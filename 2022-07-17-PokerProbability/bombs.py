from random import shuffle

import tqdm


class Poker(object):
    def __init__(self):
        self.deck = [0, 0] + [i for i in range(1, 14)] * 4

    @staticmethod
    def count_bombs(alist):
        out = 0
        if 0 in alist and alist.count(0) == 2:
            out += 1
        for i in range(1, 14):
            if i in alist and alist.count(i) == 4:
                out += 1
        return out


if __name__ == "__main__":
    print("地主摸到炸弹的概率")
    poker = Poker()
    N, n = int(1e6), 0
    for _ in tqdm.trange(N):
        shuffle(poker.deck)
        if poker.count_bombs(poker.deck[:20]):
            n += 1
    res = n / N
    print("{:.3g}".format(res))

    print("农民摸到炸弹的概率")
    poker = Poker()
    N, n = int(1e6), 0
    for _ in tqdm.trange(N):
        shuffle(poker.deck)
        if poker.count_bombs(poker.deck[:17]):
            n += 1
    res = n / N
    print("{:.3g}".format(res))
