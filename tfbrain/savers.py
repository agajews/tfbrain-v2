import json


class Saver(object):
    def __init__(self, net, fnm):
        self.net = net
        self.fnm = fnm

    def load(self):
        with open(self.fnm, 'r') as f:
            src_params = json.loads(f.read())
        self.net.set_all_params(src_params)

    def save(self):
        params = self.net.get_all_params(eval_values=True)
        params = {l_n: {p_n: p.tolist() for (p_n, p) in l_p.items()}
                  for (l_n, l_p) in params.items()}
        with open(self.fnm, 'w') as f:
            f.write(json.dumps(params))
