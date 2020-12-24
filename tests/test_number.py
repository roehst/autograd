from autograd.number import Number

def test_ols():

    import random

    rounds = 3
    lr = 0.01
    epochs = 50
    sample_size = 25

    f = open("tests/log.txt", "w")

    for _ in range(rounds):

        a_ = random.uniform(0, 1)
        b_ = random.uniform(0, 1)

        a = Number(0)
        b = Number(0)

        xs = [random.uniform(0, 1) for _ in range(sample_size)]
        es =  [random.uniform(-1, 1) for _ in range(sample_size)]

        ys = [x * b_ + a_ for x, e in zip(xs, es)]

        for epoch in range(epochs):

            for x, y in zip(xs, ys):
                
                residual = (b * x + a) - y
        
                squared_residual = (residual * residual)
                squared_residual.backward()

            a += a.grad.number * lr
            b += b.grad.number * lr

        assert abs(a.number - a_) < 0.1
        assert abs(b.number - b_) < 0.1

        f.write(f"a={a.number:0.2f}, a^={a_:0.2f}, b={b.number:0.2f}, b^={b_:0.2f}\n")

    f.close()

if __name__ == "__main__":
    test_ols()