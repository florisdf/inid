import pytest
import torch

from src.model import Recognizer, SUPPORTED_MODELS


@pytest.fixture
def device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


@pytest.mark.slow
def test_supported_models(device):
    num_classes = 5
    for model_name in SUPPORTED_MODELS:
        size = 299 if model_name == 'inception_v3' else 224
        x = torch.randn(2, 3, size, size).to(device)

        recog = Recognizer(
            model_name=model_name,
            num_classes=num_classes,
        ).to(device)

        # Should return logits during train
        recog.train()
        with torch.no_grad():
            out_train = recog(x)
        assert out_train.shape == (2, 5)

        # Should return embeddings during eval
        recog.eval()
        with torch.no_grad():
            out_eval = recog(x)
        assert out_eval.shape == (2, recog.classifier.in_features)


@pytest.fixture
def rn18_recog(device):
    return Recognizer(model_name='resnet18', num_classes=5).to(device)


def test_normalized_embeddings(rn18_recog, device):
    x = torch.randn(2, 3, 224, 224).to(device)

    rn18_recog.eval()
    with torch.no_grad():
        embs = rn18_recog(x).cpu()

    assert torch.isclose(torch.norm(embs, dim=1), torch.tensor([1., 1.])).all()


def test_normalized_weights(rn18_recog, device):
    x = torch.randn(2, 3, 224, 224).to(device)

    rn18_recog.train()
    with torch.no_grad():
        rn18_recog(x)
        ret_norm = torch.norm(rn18_recog.classifier.weight, dim=1).cpu()

    exp_norm = torch.ones(5).cpu()
    assert torch.isclose(exp_norm, ret_norm).all()


def test_bias():
    recog_bias = Recognizer(model_name='resnet18', num_classes=5,
                            clf_bias=True)
    recog_no_bias = Recognizer(model_name='resnet18', num_classes=5,
                               clf_bias=False)

    assert recog_bias.classifier.bias is not None
    assert recog_no_bias.classifier.bias is None
