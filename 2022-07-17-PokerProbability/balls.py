from math import comb


def ball_model(n, m, k):
    """return the distribution of number of marked balls chosen k balls from n
    balls. The total number of marked balls is m."""
    out = []
    partition = comb(n, k)
    for i in range(max(0, k + m - n), min(k, m) + 1):
        out.append({"X": i, "p": comb(m, i) * comb(n - m, k - i) / partition})
    return out


if __name__ == "__main__":
    res = ball_model(54, 6, 20)
    print("X | p")
    for item in res:
        print(f"{item['X']} | {item['p']:.2g}")
