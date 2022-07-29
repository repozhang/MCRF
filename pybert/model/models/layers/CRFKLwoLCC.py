from tkinter import Variable
import torch
import torch.nn as nn
from typing import List, Optional
from torch.autograd import Variable as va

class CRF(nn.Module):

    def __init__(self, num_tags: int, batch_first: bool = False) -> None:
        if num_tags <= 0:
            raise ValueError(f'invalid number of tags: {num_tags}')
        super().__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        self.transitions_prime = nn.Parameter(torch.empty(num_tags, num_tags))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions_prime, -0.1, 0.1)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(num_tags={self.num_tags})'

    def forward(self, emissions: torch.Tensor,
                tags: torch.LongTensor,
                labels_tiled, 
                mask: Optional[torch.ByteTensor] = None,
                reduction: str = 'mean') -> torch.Tensor:
        if reduction not in ('none', 'sum', 'mean', 'token_mean'):
            raise ValueError(f'invalid reduction: {reduction}')
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.uint8, device=tags.device)
        if mask.dtype != torch.uint8:
            mask = mask.byte()
        self._validate(emissions, tags=tags, mask=mask)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            mask = mask.transpose(0, 1)
            labels_tiled = labels_tiled.transpose(0, 1)

        # shape: (batch_size,)
        numerator = self._compute_score(emissions, tags, mask, labels_tiled=labels_tiled)
        # numerator = self._compute_score_(emissions, tags, mask)
        # shape: (batch_size,)
        denominator = self._compute_normalizer(emissions, mask)
        # shape: (batch_size,)
        llh = numerator + denominator
        # print(numerator, denominator)

        if reduction == 'none':
            return llh
        if reduction == 'sum':
            return llh.sum()
        if reduction == 'mean':
            return llh.mean()
        return llh.sum() / mask.float().sum()

    def decode(self, emissions: torch.Tensor,
               mask: Optional[torch.ByteTensor] = None,
               nbest: Optional[int] = None,
               pad_tag: Optional[int] = None) -> List[List[List[int]]]:
        if nbest is None:
            nbest = 1
        if mask is None:
            mask = torch.ones(emissions.shape[:2], dtype=torch.uint8,
                              device=emissions.device)
        if mask.dtype != torch.uint8:
            mask = mask.byte()
        self._validate(emissions, mask=mask)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            mask = mask.transpose(0, 1)

        if nbest == 1:
            return self._viterbi_decode(emissions, mask, pad_tag).unsqueeze(0)
        return self._viterbi_decode_nbest(emissions, mask, nbest, pad_tag)

    def _validate(self, emissions: torch.Tensor,
                  tags: Optional[torch.LongTensor] = None,
                  mask: Optional[torch.ByteTensor] = None) -> None:
        if emissions.dim() != 3:
            raise ValueError(f'emissions must have dimension of 3, got {emissions.dim()}')
        if emissions.size(2) != self.num_tags:
            raise ValueError(
                f'expected last dimension of emissions is {self.num_tags}, '
                f'got {emissions.size(2)}')

        if tags is not None:
            if emissions.shape[:2] != tags.shape:
                raise ValueError(
                    'the first two dimensions of emissions and tags must match, '
                    f'got {tuple(emissions.shape[:2])} and {tuple(tags.shape)}')

        if mask is not None:
            if emissions.shape[:2] != mask.shape:
                raise ValueError(
                    'the first two dimensions of emissions and mask must match, '
                    f'got {tuple(emissions.shape[:2])} and {tuple(mask.shape)}')
            no_empty_seq = not self.batch_first and mask[0].all()
            no_empty_seq_bf = self.batch_first and mask[:, 0].all()
            if not no_empty_seq and not no_empty_seq_bf:
                raise ValueError('mask of the first timestep must all be on')

    def _compute_score_(self, emissions: torch.Tensor,
                       tags: torch.LongTensor,
                       mask: torch.ByteTensor) -> torch.Tensor:
        seq_length, batch_size = tags.shape
        mask = mask.float()

        # Start transition score and first emission
        # shape: (batch_size,)
        score = self.start_transitions[tags[0]]
        score += emissions[0, torch.arange(batch_size), tags[0]]

        for i in range(1, seq_length):
            # Transition score to next tag, only added if next timestep is valid (mask == 1)
            # shape: (batch_size,)
            score += self.transitions[tags[i - 1], tags[i]] * mask[i]
            if i >= 2: 
                score += self.transitions_prime[tags[i - 2], tags[i]] * mask[i]

            # Emission score for next tag, only added if next timestep is valid (mask == 1)
            # shape: (batch_size,)
            score += emissions[i, torch.arange(batch_size), tags[i]] * mask[i]

        # End transition score
        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=0) - 1
        # shape: (batch_size,)
        last_tags = tags[seq_ends, torch.arange(batch_size)]
        # shape: (batch_size,)
        score += self.end_transitions[last_tags]

        return -score

    def _compute_score(self, emissions: torch.Tensor,
                       tags: torch.LongTensor,
                       mask: torch.ByteTensor, 
                       labels_tiled) -> torch.Tensor:
        # emissions: (seq_length, batch_size, num_tags)
        # tags: (seq_length, batch_size)
        # mask: (seq_length, batch_size)
        seq_length, batch_size = tags.shape
        mask = mask.float()

        # Start transition score and first emission
        # shape: (batch_size,)
        score_1 = self.start_transitions[tags[0]]
        score_2 = self.start_transitions[tags[0]]
        # print("1", score)
        # score = va(torch.zeros(batch_size, 18).to(emissions.device), requires_grad=True)
        score_1 += emissions[0, torch.arange(batch_size), tags[0]]
        score_2 += emissions[0, torch.arange(batch_size), tags[0]]
        # print("2", score)

        for i in range(1, seq_length):
            # Transition score to next tag, only added if next timestep is valid (mask == 1)
            # shape: (batch_size,)
            # score += self.transitions[tags[i - 1], tags[i]] * mask[i]
            # print("1", self.transitions[tags[i - 1], tags[i]])
            score1 = self.transitions[tags[i - 1], tags[i]] * mask[i]
            score2 = self.transitions_prime[tags[i - 1], tags[i]] * mask[i]
            # print("3", score)
            if (i >= 2): 
                # print("2", self.transitions[tags[i - 1], tags[i]])
                score2 = self.transitions_prime[tags[i - 2], tags[i]] * mask[i]
            score1 += emissions[i, torch.arange(batch_size), tags[i]] * mask[i]
            score2 += emissions[i, torch.arange(batch_size), tags[i]] * mask[i]
            score1 = labels_tiled[i, torch.arange(batch_size), tags[i]] * labels_tiled[i, torch.arange(batch_size), tags[i-1]] * score1
            # print("5", score)
            if i == 1: 
                # print("4", labels_tiled[i, torch.arange(batch_size), tags[i]] * labels_tiled[i, torch.arange(batch_size), tags[i-1]] * score)
                score2 = labels_tiled[i, torch.arange(batch_size), tags[i]] * labels_tiled[i, torch.arange(batch_size), tags[i-1]] * score2
                # print("6", score)
            else: 
                # print("5", labels_tiled[i, torch.arange(batch_size), tags[i]] * labels_tiled[i, torch.arange(batch_size), tags[i-1]] * labels_tiled[i, torch.arange(batch_size), tags[i-2]] * score)
                score2 = labels_tiled[i, torch.arange(batch_size), tags[i]] * labels_tiled[i, torch.arange(batch_size), tags[i-1]] * labels_tiled[i, torch.arange(batch_size), tags[i-2]] * score2
            score_1 += score1
            score_2 += score2
                # print("7", score)
        for i in range(seq_length): 
            score_1 -= labels_tiled[i, torch.arange(batch_size), tags[i]] * torch.log(labels_tiled[i, torch.arange(batch_size), tags[i]])
            score_2 -= labels_tiled[i, torch.arange(batch_size), tags[i]] * torch.log(labels_tiled[i, torch.arange(batch_size), tags[i]])
        score = score_1
        # End transition score
        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=0) - 1
        # shape: (batch_size,)
        last_tags = tags[seq_ends, torch.arange(batch_size)]
        # shape: (batch_size,)
        score += self.end_transitions[last_tags]

        return -score

    def _compute_normalizer(self, emissions: torch.Tensor,
                            mask: torch.ByteTensor) -> torch.Tensor:
        seq_length = emissions.size(0)
        score = self.start_transitions + emissions[0]

        for i in range(1, seq_length):
            broadcast_score = score.unsqueeze(2)
            broadcast_emissions = emissions[i].unsqueeze(1)
            next_score = broadcast_score + self.transitions + broadcast_emissions
            next_score = torch.logsumexp(next_score, dim=1)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)
        score += self.end_transitions
        return torch.logsumexp(score, dim=1)

    def _viterbi_decode(self, emissions: torch.FloatTensor,
                        mask: torch.ByteTensor,
                        pad_tag: Optional[int] = None) -> List[List[int]]:
        if pad_tag is None:
            pad_tag = 0

        device = emissions.device
        seq_length, batch_size = mask.shape
        score = self.start_transitions + emissions[0]
        history_idx = torch.zeros((seq_length, batch_size, self.num_tags),
                                  dtype=torch.long, device=device)
        oor_idx = torch.zeros((batch_size, self.num_tags),
                              dtype=torch.long, device=device)
        oor_tag = torch.full((seq_length, batch_size), pad_tag,
                             dtype=torch.long, device=device)
        for i in range(1, seq_length):
            broadcast_score = score.unsqueeze(2)
            broadcast_emission = emissions[i].unsqueeze(1)
            next_score = broadcast_score + self.transitions + broadcast_emission
            next_score, indices = next_score.max(dim=1)
            score = torch.where(mask[i].unsqueeze(-1), next_score, score)
            indices = torch.where(mask[i].unsqueeze(-1), indices, oor_idx)
            history_idx[i - 1] = indices
        end_score = score + self.end_transitions
        _, end_tag = end_score.max(dim=1)

        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=0) - 1

        history_idx = history_idx.transpose(1, 0).contiguous()
        history_idx.scatter_(1, seq_ends.view(-1, 1, 1).expand(-1, 1, self.num_tags),
                             end_tag.view(-1, 1, 1).expand(-1, 1, self.num_tags))
        history_idx = history_idx.transpose(1, 0).contiguous()

        # The most probable path for each sequence
        best_tags_arr = torch.zeros((seq_length, batch_size),
                                    dtype=torch.long, device=device)
        best_tags = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        for idx in range(seq_length - 1, -1, -1):
            best_tags = torch.gather(history_idx[idx], 1, best_tags)
            best_tags_arr[idx] = best_tags.data.view(batch_size)

        return torch.where(mask, best_tags_arr, oor_tag).transpose(0, 1)

    def _viterbi_decode_nbest(self, emissions: torch.FloatTensor,
                              mask: torch.ByteTensor,
                              nbest: int,
                              pad_tag: Optional[int] = None) -> List[List[List[int]]]:
        if pad_tag is None:
            pad_tag = 0

        device = emissions.device
        seq_length, batch_size = mask.shape

        score = self.start_transitions + emissions[0]
        history_idx = torch.zeros((seq_length, batch_size, self.num_tags, nbest),
                                  dtype=torch.long, device=device)
        oor_idx = torch.zeros((batch_size, self.num_tags, nbest),
                              dtype=torch.long, device=device)
        oor_tag = torch.full((seq_length, batch_size, nbest), pad_tag,
                             dtype=torch.long, device=device)

        for i in range(1, seq_length):
            if i == 1:
                broadcast_score = score.unsqueeze(-1)
                broadcast_emission = emissions[i].unsqueeze(1)
                # shape: (batch_size, num_tags, num_tags)
                next_score = broadcast_score + self.transitions + broadcast_emission
            else:
                broadcast_score = score.unsqueeze(-1)
                broadcast_emission = emissions[i].unsqueeze(1).unsqueeze(2)
                # shape: (batch_size, num_tags, nbest, num_tags)
                next_score = broadcast_score + self.transitions.unsqueeze(1) + broadcast_emission

            # Find the top `nbest` maximum score over all possible current tag
            # shape: (batch_size, nbest, num_tags)
            next_score, indices = next_score.view(batch_size, -1, self.num_tags).topk(nbest, dim=1)
            if i == 1:
                score = score.unsqueeze(-1).expand(-1, -1, nbest)
                indices = indices * nbest

            # convert to shape: (batch_size, num_tags, nbest)
            next_score = next_score.transpose(2, 1)
            indices = indices.transpose(2, 1)

            score = torch.where(mask[i].unsqueeze(-1).unsqueeze(-1), next_score, score)
            indices = torch.where(mask[i].unsqueeze(-1).unsqueeze(-1), indices, oor_idx)
            history_idx[i - 1] = indices

        # End transition score shape: (batch_size, num_tags, nbest)
        end_score = score + self.end_transitions.unsqueeze(-1)
        # print("end_score: ", end_score, end_score.shape)
        _, end_tag = end_score.view(batch_size, -1).topk(nbest, dim=1)
        # print(_, end_tag)

        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=0) - 1

        # insert the best tag at each sequence end (last position with mask == 1)
        history_idx = history_idx.transpose(1, 0).contiguous()
        history_idx.scatter_(1, seq_ends.view(-1, 1, 1, 1).expand(-1, 1, self.num_tags, nbest),
                             end_tag.view(-1, 1, 1, nbest).expand(-1, 1, self.num_tags, nbest))
        history_idx = history_idx.transpose(1, 0).contiguous()

        # The most probable path for each sequence
        best_tags_arr = torch.zeros((seq_length, batch_size, nbest),
                                    dtype=torch.long, device=device)
        best_tags = torch.arange(nbest, dtype=torch.long, device=device) \
                         .view(1, -1).expand(batch_size, -1)
        for idx in range(seq_length - 1, -1, -1):
            best_tags = torch.gather(history_idx[idx].view(batch_size, -1), 1, best_tags)
            best_tags_arr[idx] = best_tags.data.view(batch_size, -1) // nbest
        return torch.where(mask.unsqueeze(-1), best_tags_arr, oor_tag).permute(2, 1, 0)


if __name__ == '__main__': 
    crf = CRF(num_tags=18, batch_first=True)
    emissions = torch.rand(3,10,18)
    tags = torch.rand(3,10)
    mask = torch.ones(3,10).bool()
    print(mask)
    reduction = 'sum'
    loss = crf(emissions, tags.long(), reduction=reduction)
    print(loss)
    print(crf._viterbi_decode_nbest(emissions.permute(1,0,2), mask=mask.permute(1,0), nbest=1))


