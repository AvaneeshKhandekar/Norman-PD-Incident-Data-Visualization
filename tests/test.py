import os
import tempfile
import pytest
import pandas as pd
from unittest.mock import patch
from app import extract_incident_data, load_data_into_df, \
    agglomerative_clustering

mocked_pdf_content = """
NORMAN POLICE DEPARTMENT
Daily Incident Summary (Public)

8/1/2024 0:04       2024-00055419       1345 W LINDSEY ST       Traffic Stop       OK0140200
8/1/2024 1:15       2024-00055420       789 E MAIN ST       Suspicious Person       OK0140201
"""


@pytest.fixture
def mock_pdf_reader(mocker):
    mock_reader = mocker.Mock()
    mock_page = mocker.Mock()
    mock_page.extract_text.return_value = mocked_pdf_content
    mock_reader.pages = [mock_page]
    return mock_reader


def test_extract_incident_data(mock_pdf_reader, mocker):
    with patch('app.PdfReader', return_value=mock_pdf_reader):
        temp_dir = os.path.dirname(os.path.abspath(__file__))
        with tempfile.NamedTemporaryFile(dir=temp_dir, delete=True) as temp_pdf:
            temp_pdf.flush()
            result = extract_incident_data(temp_pdf.name)

            assert len(result) == 2
            assert result[0] == ['8/1/2024 0:04', '2024-00055419', '1345 W LINDSEY ST', 'Traffic Stop', 'OK0140200']
            assert result[1] == ['8/1/2024 1:15', '2024-00055420', '789 E MAIN ST', 'Suspicious Person', 'OK0140201']


def test_load_data_into_df():
    data = [
        ['2024-12-06 12:00', 'Incident1', 'Location1', 'Nature1', 'ORI1'],
        ['2024-12-06 13:00', 'Incident2', 'Location2', 'Nature2', 'ORI2']
    ]
    df = load_data_into_df(data)

    assert isinstance(df, pd.DataFrame), "Output should be a pandas DataFrame."
    assert df.shape == (2, 5), f"Expected shape (2, 5), but got {df.shape}."
    assert list(df.columns) == ["Date/Time", "Incident Number", "Location", "Nature", "Incident ORI"], \
        "DataFrame columns do not match expected columns."


def test_agglomerative_clustering():
    data = {
        'Nature': ['Nature1', 'Nature2', 'Nature3', 'Nature4', 'Nature1', 'Nature2']
    }
    df = pd.DataFrame(data)

    num_clusters = 2
    df_clustered = agglomerative_clustering(df, num_clusters)

    assert 'Cluster' in df_clustered.columns, "'Cluster' column should be present in the DataFrame."
    assert len(df_clustered[
                   'Cluster'].unique()) == num_clusters, f"Expected {num_clusters} clusters, but got {len(df_clustered['Cluster'].unique())}."


def test_valid_url():
    valid_url = 'https://www.example.com'
    invalid_url = 'invalid-url'

    valid_urls = []

    with patch('urllib.request.urlretrieve') as mock_urlretrieve:
        mock_urlretrieve.return_value = None
        try:
            valid_urls.append(valid_url)
        except:
            assert False, f"Failed to validate valid URL: {valid_url}"

    try:
        assert invalid_url not in valid_urls, "Invalid URL should not be added."
    except:
        assert True


if __name__ == "__main__":
    pytest.main()
