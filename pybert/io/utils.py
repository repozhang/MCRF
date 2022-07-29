import torch
def collate_fn(batch):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    input_lens, num, num1, num2, turn, input_ids, input_mask, segment_ids, pos_ids, labels, multi_labels, output_mask, all_output_mask = map(torch.stack, zip(*batch))
    # max_len = max(input_lens).item()
    # input_ids = input_ids[:, :max_len]
    # input_mask = input_mask[:, :max_len]
    # output_mask = output_mask[:, :max_len]
    # all_output_mask = all_output_mask[:, :max_len]
    # segment_ids = segment_ids[:, :max_len]
    return input_lens, num, num1, num2, turn, input_ids, input_mask, segment_ids, pos_ids, labels, multi_labels, output_mask, all_output_mask 


