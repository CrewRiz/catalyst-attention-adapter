from catalyst_attention_adapter import CatalystSoftmaxAttention, encode_label


dim = 256
attention = CatalystSoftmaxAttention(dim=dim, nqubits=4)

key_labels = ["billing", "repo", "tests", "docs"]
keys = [encode_label(label, dim) for label in key_labels]
values = [encode_label(f"value:{label}", dim) for label in key_labels]

result = attention.forward_with_metadata(keys[2], keys, values)

print(result.selected_index)
print(result.confidence)
print(len(result.output))
