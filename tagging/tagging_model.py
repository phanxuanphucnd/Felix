import torch
from transformers import AutoConfig, AutoModel, RobertaTokenizer
import torch.nn as nn
from tagging.positional_embedding import PositionEmbedding
import math


class Biaffine(nn.Module):
    r"""
    Biaffine layer for first-order scoring :cite:`dozat-etal-2017-biaffine`.
    This function has a tensor of weights :math:`W` and bias terms if needed.
    The score :math:`s(x, y)` of the vector pair :math:`(x, y)` is computed as :math:`x^T W y / d^s`,
    where `d` and `s` are vector dimension and scaling factor respectively.
    :math:`x` and :math:`y` can be concatenated with bias terms.
    Args:
        n_in (int):
            The size of the input feature.
        n_out (int):
            The number of output channels.
        scale (float):
            Factor to scale the scores. Default: 0.
        bias_x (bool):
            If ``True``, adds a bias term for tensor :math:`x`. Default: ``True``.
        bias_y (bool):
            If ``True``, adds a bias term for tensor :math:`y`. Default: ``True``.
    """

    def __init__(self, n_in, n_out=1, scale=0, bias_x=True, bias_y=True):
        super().__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.scale = scale
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.weight = nn.Parameter(torch.Tensor(n_out, n_in+bias_x, n_in+bias_y))

        self.reset_parameters()

    def __repr__(self):
        s = f"n_in={self.n_in}"
        if self.n_out > 1:
            s += f", n_out={self.n_out}"
        if self.scale != 0:
            s += f", scale={self.scale}"
        if self.bias_x:
            s += f", bias_x={self.bias_x}"
        if self.bias_y:
            s += f", bias_y={self.bias_y}"

        return f"{self.__class__.__name__}({s})"

    def reset_parameters(self):
        nn.init.zeros_(self.weight)

    def forward(self, x, y):
        r"""
        Args:
            x (torch.Tensor): ``[batch_size, seq_len, n_in]``.
            y (torch.Tensor): ``[batch_size, seq_len, n_in]``.
        Returns:
            ~torch.Tensor:
                A scoring tensor of shape ``[batch_size, n_out, seq_len, seq_len]``.
                If ``n_out=1``, the dimension for ``n_out`` will be squeezed automatically.
        """

        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
        # [batch_size, n_out, seq_len, seq_len]
        s = torch.einsum('bxi,oij,byj->boxy', x, self.weight, y) / self.n_in ** self.scale
        # remove dim 1 if n_out == 1
        s = s.squeeze(1)

        return s


class SelfAttentionMask(nn.Module):
    """Create 3D attention mask from a 2D tensor mask.
      inputs[0]: from_tensor: 2D or 3D Tensor of shape
        [batch_size, from_seq_length, ...].
      inputs[1]: to_mask: int32 Tensor of shape [batch_size, to_seq_length].
      Returns:
        float Tensor of shape [batch_size, from_seq_length, to_seq_length].
    """
    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, inputs, to_mask=None):
        if isinstance(inputs, list) and to_mask is None:
            to_mask = inputs[1]
            inputs = inputs[0]
        from_shape = inputs.size()
        batch_size = from_shape[0]
        from_seq_length = from_shape[1]

        to_shape = to_mask.size()
        to_seq_length = to_shape[1]

        to_mask = to_mask.unsqueeze(1).type(inputs.dtype)
        # to_mask = tf.cast(
        #     tf.reshape(to_mask, [batch_size, 1, to_seq_length]),
        #     dtype=inputs.dtype)

        # We don't assume that `from_tensor` is a mask (although it could be). We
        # don't actually care if we attend *from* padding tokens (only *to* padding)
        # tokens so we create a tensor of all ones.
        #
        # `broadcast_ones` = [batch_size, from_seq_length, 1]
        broadcast_ones = torch.ones(
            (batch_size, from_seq_length, 1), dtype=inputs.dtype).to(self.device)

        # Here we broadcast along two dimensions to create the mask.
        mask = broadcast_ones * to_mask

        return mask


class FelixTagger(nn.Module):
    def __init__(self,
                 model_name="vinai/phobert-base",
                 max_word_length=128,
                 use_pointing=True,
                 num_classes=14,
                 position_embedding_dim=64,
                 query_dim=6,
                 query_transformer=2,
                 device='cuda',
                 is_training=True):
        super().__init__()
        self._is_training = is_training
        self.model_name = model_name
        self.query_transformer = query_transformer
        self.device = device
        self.config = AutoConfig.from_pretrained(
            self.model_name, from_tf=False, output_hidden_states=True
        )
        self.max_word_length = max_word_length
        self.num_classes = num_classes
        self.tag_embedding_dim = int(math.ceil(math.sqrt(self.num_classes)))
        self.tag_embedding_dim += 1 if self.tag_embedding_dim % 2 != 0 else 0
            

        self.biaffine = Biaffine(query_dim)
        self.roberta = AutoModel.from_pretrained(
            self.model_name, config=self.config)
        self._tag_logits_layer = nn.Linear(
            self.config.hidden_size, self.num_classes)
        self._tag_embedding_layer = nn.Embedding(
            self.num_classes, self.tag_embedding_dim)
        self._positional_embeddings_layer = PositionEmbedding(
            self.tag_embedding_dim)
        self._edit_tagged_sequence_output_layer = nn.Linear(
            self.config.hidden_size+self.tag_embedding_dim*2, self.config.hidden_size)
        self.activation_fn = nn.GELU()

        if self.query_transformer:
            self._self_attention_mask_layer = SelfAttentionMask(self.device)
            self._transformer_query_layer = nn.TransformerEncoderLayer(
                d_model=self.config.hidden_size,
                nhead=self.config.num_attention_heads,
                dim_feedforward=self.config.intermediate_size,
                activation=self.config.hidden_act,
                dropout=self.config.hidden_dropout_prob,
                batch_first=True,
            )

        self._query_embeddings_layer = nn.Linear(self.config.hidden_size, query_dim)
        self._key_embeddings_layer = nn.Linear(self.config.hidden_size, query_dim)

    def agg_bpe2word(self, bpe_embeddings, word_bpe_matrix, mode="sum"):
        word_embeddings = torch.bmm(word_bpe_matrix, bpe_embeddings)
        if mode == "sum":
            return word_embeddings
        elif mode == "mean":
            d_n = word_bpe_matrix.sum(dim=-1).unsqueeze(-1)
            d_n[d_n == 0] = 1
            return word_embeddings / d_n

    def _attention_scores(self, query, key, mask=None, device="cuda"):
        """Calculates attention scores as a query-key dot product.
        Args:
        query: Query tensor of shape `[batch_size, sequence_length, Tq]`.
        key: Key tensor of shape `[batch_size, sequence_length, Tv]`.
        mask: mask tensor of shape `[batch_size, sequence_length]`.
        Returns:
        Tensor of shape `[batch_size, sequence_length, sequence_length]`.
        """
        scores = torch.matmul(query, key.transpose(1, 2))
        # scores = self.biaffine(query, key)
        # print("scores", scores.size())

        if mask is not None:
            mask = SelfAttentionMask(self.device)(scores, mask)
            # print(mask)
            # print("mask", mask.size()) [batch_size, sequence_length, sequence_length]
            # Prevent pointing to self (zeros down the diagonal).
            # print((mask.size(0), self.max_word_length))
            a = torch.ones((mask.size(0), self.max_word_length)).to(device)
            a = torch.diag_embed(a)
            diagonal_mask = 1-a
            # print(diagonal_mask)
            diagonal_mask = diagonal_mask.type(torch.float32)
            # diagonal_mask = tf.linalg.diag(
            #     tf.zeros((tf.shape(mask)[0], self.max_word_length)), padding_value=1)
            # diagonal_mask = tf.cast(diagonal_mask, tf.float32)
            mask = diagonal_mask * mask
            # print("mask", mask.size())
            # As this is pre softmax (exp) as such we set the values very low.
            mask_add = -1e9 * (1. - mask)
            # print(mask)
            # exit()
            scores = scores * mask + mask_add

        return scores

    def forward(
        self,
        inputs=None
    ):
        """Forward pass of the model.
        Args:
        inputs:
            A list of tensors. In training the following 4 tensors are required,
            [input_word_ids, input_mask, input_type_ids, edit_tags]. Only the first 3
            are required in test. input_word_ids[batch_size, seq_length],
            input_mask[batch_size, seq_length], input_type_ids[batch_size,
            seq_length], edit_tags[batch_size, seq_length]. If using output
            variants, these should also be provided. output_variant_ids[batch_size,
            1].
        Returns:
        The logits of the edit tags and optionally the logits of the pointer
            network.
        """
        if self._is_training:
            input_ids, edit_tags, point_tags, word_matrix, attention_mask, input_mask_words = inputs
        else:
            input_ids, word_matrix, attention_mask, input_mask_words = inputs
        # print("input_mask_words", input_mask_words.size()) [batch_sz, seq_len_words]

        bert_output = self.roberta(input_ids, attention_mask)[0]
        # print("bert_output", bert_output.size()) # [batch_sz, seq_len_subwords, hidden_sz]
        bert_output = self.agg_bpe2word(
            bert_output, word_matrix, "sum")
        # print("bert_output", bert_output.size()) # [batch_sz, seq_len_words, hidden_sz]
        tag_logits = self._tag_logits_layer(bert_output)
        # print(tag_logits.size()) # [batch_sz, seq_len, num_class]
        if not self._is_training:
            edit_tags = torch.argmax(tag_logits, dim=-1)

        tag_embedding = self._tag_embedding_layer(edit_tags)
        # print("tag_embedding", tag_embedding.size()) # [batch_sz, seq_len, tag_embedding_dim]
        position_embedding = self._positional_embeddings_layer(
            tag_embedding.transpose(1, 0))
        position_embedding = position_embedding.transpose(1, 0)
        # print("position_embedding", position_embedding.size()) # [batch_sz, seq_len, tag_embedding_dim]
        edit_tagged_sequence_output = self._edit_tagged_sequence_output_layer(
            torch.cat((
                bert_output, tag_embedding, position_embedding), -1)
        )
        # print("edit_tagged_sequence_output", edit_tagged_sequence_output.size()) # [batch_sz, seq_len, hidden_size]
        intermediate_query_embeddings = edit_tagged_sequence_output
        if self.query_transformer:
            src_mask = self._self_attention_mask_layer(
                intermediate_query_embeddings, input_mask_words)
            src_mask = src_mask.repeat(self.config.num_attention_heads, 1, 1, 1)
            src_mask = src_mask.view(-1, src_mask.size(-1), src_mask.size(-1))
            for _ in range(int(self.query_transformer)):
                intermediate_query_embeddings = self._transformer_query_layer(
                    intermediate_query_embeddings, src_mask)
        # print(intermediate_query_embeddings.size()) # [batch_sz, seq_len, hidden_size]
        query_embeddings = self._query_embeddings_layer(
            intermediate_query_embeddings)
        # print(query_embeddings.size()) # [batch_sz, seq_len, query_dim]
        key_embeddings = self._key_embeddings_layer(
            edit_tagged_sequence_output)
        # print(key_embeddings.size()) # [batch_sz, seq_len, query_dim]
        # ,tf.cast(input_mask, tf.float32))
        pointing_logits = self._attention_scores(
            query_embeddings, key_embeddings, input_mask_words, self.device)
        # print(pointing_logits.size()) # [batch_sz, seq_len, seq_len]
        return tag_logits, pointing_logits


if __name__ == "__main__":
    model_path = "../shared_data/BDIRoBerta"
    device = "cuda"
    model = FelixTagger(
        model_name="../shared_data/BDIRoBerta")
    model.to(device)
    tokenizer = RobertaTokenizer.from_pretrained(model_path)
    print(tokenizer.bos_token)
    print(tokenizer.eos_token)
    exit()
    sentence = "Loài tôm kiếm ăn vào lúc nào?"
    tokens = tokenizer.tokenize(sentence)
    print(tokens)
    exit()
    input_ids = tokenizer.encode(sentence, return_tensors="pt").to(device)
    edit_ids = torch.tensor(
        [2, 2, 2, 2, 2, 2, 3, 3, 3, 2, 2, 2], dtype=torch.long, device=device)
    edit_ids = torch.unsqueeze(edit_ids, 0)
    print(input_ids)
    print(edit_ids)
    inputs = (input_ids, None, None, edit_ids)
    tag_logits, pointing_logits = model(inputs)
    print(tag_logits, pointing_logits)