import torch
import numpy as np
import math
import fcos_core.deepshift.kernels

def round_to_fixed(input, fraction=16, integer=16):
    assert integer >= 1, integer 
    if integer == 1: 
        return torch.sign(input) - 1 
    delta = math.pow(2.0, -(fraction))
    bound = math.pow(2.0, integer-1) 
    min_val = - bound 
    max_val = bound - 1 
    rounded = torch.floor(input / delta) * delta

    clipped_value = torch.clamp(rounded, min_val, max_val)
    return clipped_value 

def get_shift_and_sign(x, rounding='deterministic'):
    '''分别返回P和S'''
    sign = torch.sign(x)
    
    x_abs = torch.abs(x)
    shift = round(torch.log(x_abs) / np.log(2), rounding)
    
    return shift, sign

def round_power_of_2(x, rounding='deterministic'):
    '''W = S * 2^P'''
    shift, sign = get_shift_and_sign(x, rounding)
    x_rounded = (2.0 ** shift) * sign
    return x_rounded

def round(x, rounding='deterministic'):
    assert(rounding in ['deterministic', 'stochastic'])
    if rounding == 'stochastic':
        x_floor = x.floor()
        return x_floor + torch.bernoulli(x - x_floor)
    else:
        return x.round()

class ConcWeight():
    def __init__(self, data=None, base=0, bits=8):
        self.data = data 
        self.base = base
        self.bits = bits

##concatenate shift and sign together
def compress_bits(shift, sign):
    conc_weight = ConcWeight() 

    if len(shift.shape) == 2:
        shift = shift.unsqueeze(-1).unsqueeze(-1)

    # if sign is ternary, then use a big shift value that is equivalent to multiplying by zero
    # w * 0 ~= (+)(w >> 32)
    zero_sign_indices = (sign == 0).nonzero()
    shift[zero_sign_indices] = -32
    sign[zero_sign_indices] = +1

    conc_weight.bits = math.ceil(torch.log( - torch.min(shift) + 1)/ np.log(2))
    # treat shift to the right as the default
    shift = shift * -1
    # 最小是原来的最大值的负值
    minimum = int(torch.min(shift))
    if minimum < 0:
        conc_weight.base = minimum
        shift = shift - minimum
    else:
        conc_weight.base = 0

    num = int(32 / (conc_weight.bits + 1))
    row_length = int((shift.shape[1] * shift.shape[2] * shift.shape[3] + num -1) / num )
    size = row_length * shift.shape[0]

    conc_weight.data = fcos_core.deepshift.kernels.compress_sign_and_shift(shift.int().cuda(), sign.int().cuda(), size, conc_weight.base, conc_weight.bits, row_length, num)

    return conc_weight


if __name__ == "__main__":
    datas = [
        torch.tensor(0.1),
        torch.tensor(0.5),
        torch.tensor(0.9),
        torch.tensor(1.5),
        torch.tensor(2.0),
        torch.tensor(3.0),
        torch.tensor(-0.1),
        torch.tensor(-0.5),
        torch.tensor(-0.9),
        torch.tensor(-1.5),
        torch.tensor(-2.0),
        torch.tensor(-3.0),
        torch.tensor(0.12345678),
    ]
    for d in datas:
        r1 = round_power_of_2(d)
        r2 = round_to_fixed(d, fraction=4)
        print(d, r1, r2)
# tensor(0.1000) tensor(0.1250) tensor(0.0625)
# tensor(0.5000) tensor(0.5000) tensor(0.5000)
# tensor(0.9000) tensor(1.) tensor(0.8750)
# tensor(1.5000) tensor(2.) tensor(1.5000)
# tensor(2.) tensor(2.) tensor(2.)
# tensor(3.) tensor(4.) tensor(3.)
# tensor(-0.1000) tensor(-0.1250) tensor(-0.1250)
# tensor(-0.5000) tensor(-0.5000) tensor(-0.5000)
# tensor(-0.9000) tensor(-1.) tensor(-0.9375)
# tensor(-1.5000) tensor(-2.) tensor(-1.5000)
# tensor(-2.) tensor(-2.) tensor(-2.)
# tensor(-3.) tensor(-4.) tensor(-3.)
# tensor(0.1235) tensor(0.1250) tensor(0.0625)
# round_power_of_2是将权值量化到2^n范围内
# round_to_fixed是将权值量化到二进制n位小数上，如果fraction=16，实际上16位二进制小数位精度也很高了，不比浮点低
    shift = torch.tensor([[[[-5., -4.], [-6., 3.]], [[7., 2.], [12., 8.]]]])
    sign = torch.tensor([[[[-1., 1.], [1., -1.]], [[1., 1.], [-1., -1.]]]])
    conc_weight = compress_bits(shift, sign)
    print(conc_weight)
# base=-12, bits=3, data=tensor([234446813])