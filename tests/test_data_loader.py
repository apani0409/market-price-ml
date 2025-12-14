from src.data_loader import load_data

def test_data_loading():
    data = load_data()  # Use actual data path or mock
    assert data is not None
    assert not data.empty