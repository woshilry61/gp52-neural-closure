"""CUDA graph of a fixed-shape loss and gradient, reusable by L-BFGS."""
import torch
def capture_backward(model):
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            model.zero_grad(set_to_none=True)
            warm=model.loss();warm.backward()
        del warm
    torch.cuda.current_stream().wait_stream(stream)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):
        model.zero_grad(set_to_none=False)
        loss=model.loss();loss.backward()
    torch.cuda.current_stream().wait_stream(stream)
    def closure():
        graph.replay()
        return loss
    return closure
