from chesscontroller import load_best_conf, simulate_many

conf = load_best_conf()
simulate_many(100, conf, conf, True)
