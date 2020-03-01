from catalyst.dl import registry, Callback, CallbackOrder, State


@registry.Callback
class MNISTInferCallback(Callback):

    def __init__(self, subm_file):
        super().__init__(CallbackOrder.Internal)
        self.subm_file = subm_file
        self.preds = []

    def on_batch_end(self, state: State):
        paths = state.input["paths"]
        preds = state.output["logits"].detach().cpu().numpy()
        preds = preds.argmax(axis=1)
        for path, pred in zip(paths, preds):
            self.preds.append((path, pred))

    def on_loader_end(self, _):
        subm = ["path,class_id"]
        subm += [f"{path},{cls}" for path, cls in self.preds]
        with open(self.subm_file, 'w') as file:
            file.write("\n".join(subm))
