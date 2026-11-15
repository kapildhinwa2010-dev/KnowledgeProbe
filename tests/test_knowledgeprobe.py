import pytest
import torch
from knowledgeprobe import KnowledgeProbe, create_knowledgeprobe

class TestKnowledgeProbe:
    def test_init(self):
        m = KnowledgeProbe(dim=256, hidden_dim=512)
        assert m.dim == 256
    
    def test_forward(self):
        m = KnowledgeProbe(dim=256)
        x = torch.randn(32, 256)
        assert m(x).shape == (32, 256)
    
    def test_factory(self):
        m = create_knowledgeprobe()
        assert isinstance(m, KnowledgeProbe)

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
