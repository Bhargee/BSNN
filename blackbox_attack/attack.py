from argparse import ArgumentParser
import random 
import time

import torch 
from torchvision.utils import save_image

from .data import cifar10
from models import resnet


def _attack(model, dataset, x0, y0, alpha=.2, beta=.001, iterations=1000):
    if model.predict(x0) != y0:
        print('incorrect classification, no need to attack')
        return x0

    num_samples = 1000
    best_theta, g_theta = None, float('inf')
    query_count = 0

    print(f'searching for initial direction on {num_samples} samples')
    samples = set(random.sample(range(len(dataset)), num_samples))
    timestart = time.time()

    for i, (xi, yi) in enumerate(dataset):
        if i not in samples:
            continue
        query_count += 1
        if model.predict(xi) != y0:
            theta = xi - x0
            initial_lbd = torch.norm(theta)
            theta = theta/torch.norm(theta)
            lbd, count = fine_grained_binary_search(model, x0, y0, theta,
                    initial_lbd, g_theta)
            query_count += count
            if lbd < g_theta:
                best_theta, g_theta = theta, lbd
                print(f'distortion: {g_theta}')
    timeend = time.time()
    print(f'best distortion: {g_theta},({timeend-timestart} s, {query_count} q)')

    timestart = time.time()
    g1 = 1.0
    theta, g2 = best_theta.clone(), g_theta
    torch.manual_seed(0)
    opt_count = 0
    stopping = .01
    prev_obj = 100000
    for i in range(iterations):
        gradient = torch.zeros(theta.size())
        q = 10
        min_g1 = float('inf')
        for _ in range(q):
            u = torch.randn(theta.size()).float()
            u = u/torch.norm(u)
            ttt = theta + beta * u
            ttt = ttt/torch.norm(ttt)
            g1, count = fine_grained_binary_search_local(model, x0, y0, ttt,
                    initial_lbd=g2, tol=beta/500)
            opt_count += count
            gradient += (g1-g2)/beta * u
            if g1 < min_g1:
                min_g1 = g1
                min_ttt = ttt
        gradient = 1.0/q * gradient

        if (i+1)%50 == 0:
            print(f'iter {i+1}: g(theta+beta*u)={g1}, g(theta)={g2}, distortion={torch.norm(g2*theta)}, num_queries={opt_count}')
            if g2 > prev_obj - stopping:
                break
            prev_obj = g2

        min_theta = theta
        min_g2 = g2

        for _ in range(15):
            new_theta = theta - alpha*gradient
            new_theta = new_theta/torch.norm(new_theta)
            new_g2, count = fine_grained_binary_search_local(model, x0, y0,
                    new_theta, initial_lbd=min_g2, tol=beta/500)
            opt_count += count
            alpha = alpha * 2
            if new_g2 < min_g2:
                min_theta = new_theta
                min_g2 = new_g2
            else:
                break

        if min_g2 >= g2:
            for _ in range(15):
                alpha = alpha * .25
                new_theta = theta - alpha * gradient
                new_theta = new_theta/torch.norm(new_theta)
                new_g2, count = fine_grained_binary_search_local(model, x0, y0,
                        new_theta, initial_lbd = min_g2, tol=beta/500)
                opt_count += count
                if new_g2 < g2:
                    min_theta = new_theta 
                    min_g2 = new_g2
                    break

        if min_g2 <= min_g1:
            theta, g2 = min_theta, min_g2
        else:
            theta, g2 = min_ttt, min_g1

        if g2 < g_theta:
            best_theta, g_theta = theta.clone(), g2
        
        if alpha < 1e-4:
            alpha = 1.0
            print(f'Warning: not moving, g2 {g2} gtheta {g_theta}')
            beta = beta * 0.1
            if (beta < 0.0005):
                break

    target = model.predict(x0 + g_theta*best_theta)
    timeend = time.time()
    print(f'\nfound adversarial example: distortion %.4f target %d queries %d \nTime: %.4f seconds' % (g_theta, target, query_count + opt_count, timeend-timestart))
    return x0 + g_theta*best_theta



def fine_grained_binary_search(model, x0, y0, theta, initial_lbd, current_best):
    nquery = 0
    if initial_lbd > current_best:
        if model.predict(x0+current_best*theta) == y0:
            nquery += 1
            return float('inf'), nquery
        lbd = current_best
    else:
        lbd = initial_lbd

    lbd_hi = lbd
    lbd_lo = 0.

    while (lbd_hi - lbd_lo) > 1e-5:
        lbd_mid = (lbd_lo + lbd_hi)/2.
        nquery += 1
        if model.predict(x0 + lbd_mid*theta) != y0:
            lbd_hi = lbd_mid
        else:
            lbd_lo = lbd_mid
    return lbd_hi, nquery


def fine_grained_binary_search_local(model, x0, y0, theta, initial_lbd=1.0, tol=1e-5):
    nquery = 0
    lbd = initial_lbd

    if model.predict(x0+lbd*theta) == y0:
        lbd_lo = lbd
        lbd_hi = lbd*1.01
        nquery += 1
        while model.predict(x0+lbd_hi*theta) == y0:
            lbd_hi = lbd_hi*1.01
            nquery += 1
            if lbd_hi > 20:
                return float('inf'), nquery
    else:
        lbd_hi = lbd
        lbd_lo = lbd*.99
        nquery += 1
        while model.predict(x0+lbd_lo*theta) != y0:
            lbd_lo = lbd_lo*.99
            nquery += 1

    while (lbd_hi - lbd_lo) > tol:
        lbd_mid = (lbd_lo + lbd_hi)/2.
        nquery += 1
        if model.predict(x0 + lbd_mid*theta) != y0:
            lbd_hi = lbd_mid
        else:
            lbd_lo = lbd_mid
    return lbd_hi, nquery


def attack_cifar10(model, alpha=.2, beta=.001, num_attacks=100):
    model.eval()
    train, test = cifar10()
    print(f'length of test set: {len(test)}')

    def single_attack(image, label):
        print(f'original label: {label}')
        print(f'predicted label: {model.predict(image)}')
        adversarial = _attack(model, train, image, label, alpha=alpha,
                             beta=beta, iterations=1000)
        save_image(image, 'original.png')
        save_image(adversarial, 'adversarial.png')
        return torch.norm(adversarial-image)

    print(f'running attck on {num_attacks} random CIFAR10 test images')
    print(f'alpha={alpha}, beta={beta}')
    total_distortion = 0.

    samples = [6311, 6890, 663, 4242, 8376, 7961, 6634, 4969, 7808, 5866, 9558,
            3578, 8268, 2281, 2289, 1553, 4104, 8725, 9861, 2407, 5081, 1618,
            1208, 5409, 7735, 9171, 1649, 5796, 7113, 5180, 3350,9052, 7253,
            8541, 4267, 1020, 8989, 230, 1528, 6534, 18, 8086, 3996, 1031,
            3130, 9298, 3632, 3909, 2334, 8896, 7339, 1494, 5243, 8322, 8016,
            1786, 9031, 4769, 8969, 5451, 8852, 3329, 9882, 8965, 9627, 4712,
            7290, 9769, 6306, 5194, 3966, 4756, 3012, 3102, 540, 4260, 7807,
            1471, 2133, 2450, 633, 1314, 8857, 6410, 8594, 4515, 8549, 3858,
            3525, 6411, 4360, 7753, 7413, 684,3343, 6785, 7079, 2263]
    samples = [4242] # TODO remove
    for idx in samples:
        image, label = test[idx]
        image = image.float()
        print(f'image {idx}')
        total_distortion += single_attack(image, label)

    avg_distortion = total_distortion/num_attacks
    print(f'average distortion on {num_attacks} images: {avg_distortion}')


def main():
    assert torch.cuda.is_available()
    p = ArgumentParser('Opt black box attack')
    p.add_argument('model_class', help='resnet<n1n2>, for example')
    p.add_argument('saved_params', help='saved weights for model_class')
    # TODO below currently doesn't do anything, will need to change
    p.add_argument('--stochastic', required=False, help='use BSNN model')
    args = p.parse_args()

    constructor = getattr(resnet, args.model_class)
    model = constructor()
    model.load_state_dict(torch.load(args.saved_params))
    with torch.no_grad():
        attack_cifar10(model, alpha=5., beta=.001)


if __name__ == '__main__':
    main()
