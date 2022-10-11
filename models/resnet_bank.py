import torch
from .resnet import ResNet, BasicBlock, Bottleneck


class ResNetBank(ResNet):

    def __init__(self,
                 K,
                 *args,
                 **kwargs):
        super(ResNetBank, self).__init__(*args, **kwargs)
        # create the queue
        self.K = K
        self.register_buffer("queue", torch.randn(self.num_classes, K, self.out_channel))
        self.register_buffer("queue_ptr", torch.zeros(self.num_classes, 1, dtype=torch.long))

    @torch.no_grad()
    def dequeue_and_enqueue(self, keys, targets):
        # gather keys before updating queue
        targets_unique = torch.unique(targets)
        for i in targets_unique:
            cls_inds = (targets == i)
            if cls_inds.sum() == 0:
                continue
            _keys = keys[cls_inds, :]
            batch_size = _keys.shape[0]
            ptr = int(self.queue_ptr[i])

            # replace the keys at ptr (dequeue and enqueue)
            if ptr + batch_size <= self.K:
                self.queue[i, ptr:ptr + batch_size, :] = _keys
            else:
                self.queue[i, ptr:, :] = _keys[:(self.K - ptr)]
                _ptr = (ptr + batch_size) % self.K
                self.queue[i, :_ptr, :] = _keys[(self.K - ptr):]

            ptr = (ptr + batch_size) % self.K  # move pointer

            self.queue_ptr[i, 0] = ptr

def _resnet_bank(arch, K, block, layers, pretrained, progress, **kwargs):
    model = ResNetBank(K, block, layers, **kwargs)
    return model


def resnet18_bank(K, pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet_bank('resnet18', K, BasicBlock, [2, 2, 2, 2], pretrained, progress,
                        **kwargs)


def resnet50_bank(K, pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet_bank('resnet50', K, Bottleneck, [3, 4, 6, 3], pretrained, progress,
                        **kwargs)
