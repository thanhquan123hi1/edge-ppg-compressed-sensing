"""Capture an unchanged FP32 Adam step; restore warmup weights, moments and BN."""
import copy
import torch

class CapturedAdamStep:
    def __init__(self,model,optimizer,y,x):
        self.model=model;self.optimizer=optimizer;model.train()
        self.y=y.clone();self.x=x.clone()
        original={k:v.detach().clone() for k,v in model.state_dict().items()}
        original_opt={p:{k:v.clone() if torch.is_tensor(v) else copy.deepcopy(v) for k,v in state.items()} for p,state in optimizer.state.items()}
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                optimizer.zero_grad(set_to_none=True);loss=(model(self.y)-self.x).square().mean();loss.backward();optimizer.step()
        torch.cuda.current_stream().wait_stream(stream)
        model.load_state_dict(original)
        for p,state in optimizer.state.items():
            for k,v in state.items():
                if torch.is_tensor(v):
                    if p in original_opt:v.copy_(original_opt[p][k])
                    else:v.zero_()
        optimizer.zero_grad(set_to_none=True);self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.loss=(model(self.y)-self.x).square().mean();self.loss.backward();optimizer.step()

    def step(self,y,x):
        if y.shape!=self.y.shape:
            self.optimizer.zero_grad(set_to_none=True)
            loss=(self.model(y)-x).square().mean();loss.backward();self.optimizer.step();return loss
        self.y.copy_(y);self.x.copy_(x);self.graph.replay();return self.loss
