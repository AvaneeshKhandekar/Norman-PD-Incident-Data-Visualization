# Norman PD Incident Data Visualization (The End Pipeline)

### AUTHOR

##### Avaneesh Khandekar

### INSTALLATION

To Install required dependencies:

``` bash
pipenv install
```

### USAGE

To fetch incident data from a specified PDF report URL:

```bash
pipenv run streamlit run app.py
```

Visit: ```http://localhost:8501/``` to access the app.

## Demo

[Click here to watch the demo video](https://uflorida-my.sharepoint.com/:v:/g/personal/akhandekar_ufl_edu/EeOsaWxGhtdCojfZT-uH15gBWW9xmtViyvamubwbLO6JBw?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJPbmVEcml2ZUZvckJ1c2luZXNzIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXciLCJyZWZlcnJhbFZpZXciOiJNeUZpbGVzTGlua0NvcHkifX0&e=DpDaev)

### OVERVIEW

This project extracts and visualizes incident data from the Norman Police Department's Incident Summary reports in PDF
format.
It provides insights into incident patterns and trends using clustering and data visualization techniques.
For the User Interface, Streamlit is used.
Streamlit is an open-source Python framework for data scientists and AI/ML engineers to deliver interactive data apps.

#### Columns available in the PDF:

- Date/Time: The date and time of the incident.
- Incident Number: A unique identifier for the incident.
- Location: The address where the incident occurred.
- Nature: The type of incident (e.g., Traffic Stop).
- Incident ORI: The Originating Agency Identifier.

### Features:

#### PDF Data Extraction:

- Extracts incident data (e.g., Date/Time, Incident Number, Location, Nature, Incident ORI) from uploaded PDF files
  using `PyPDF`.

#### Data Clustering:

- Clusters incident descriptions (Nature) based on semantic similarity using `SentenceTransformer` and
  `Agglomerative Clustering`.

#### Visualization:

- `Scatter plot` of clusters reduced to 2D using `PCA`.
- `Bar chart` for the frequency of incidents by type `(Nature)`.
- `Line chart` showing `hourly trends` of incidents.

#### Feedback Section:

- Collects user feedback to improve the tool.
- Currently the feedback form is not submitted anywhere, but we can integrate an SMTP server or a database to send the
  feedback via mail or store in a database.

### Workflow

- User has 2 options to submit files.
    - Use the Upload NormanPD Incident Summary PDFs widget to upload PDF files from your local system.
        - The code checks if the uploaded PDFs are valid otherwise throws an error saying invalid PDF.
    - Paste URLs of the PDFs in the provided text area (comma-separated).
        - The code checks if the URL retrieves valid PDFs otherwise throws an error saying invalid URL.
- They can be combined meaning user can upload files and also provide URLs and all of the data will be considered. To
  reset just refresh the page.
- Data is parsed, cleaned and extracted text to create a structured dataset.
    - Start reading PDF in layout mode (This preserves the spaces in original pdf).
        - Skip blank lines.
        - Split all text line by line.
        - Iterate over lines:
            - Check if it is a header or default line and continue.
            - Split the lines based on large whitespace. If the space is more than 5 spaces then consider it as one
              column.
            - Check record length and add it as a list to result array.
            - If the record length is less than 5 then this is continuation of the location part on the next line. In
              this
              case update location part of the last added list in the result array.
- The resultant array is then converted into a Pandas DataFrame.
- **Raw Data Table:** Displays extracted data in a structured tabular format. It is interactive ad can be saved as a CSV
  and
  can be searched for specific data.
- **Clustering Visualization:** Groups incidents based on the semantic similarity of their "Nature" field.
    - Clusters incident descriptions into groups based on semantic similarity.
    - Agglomerative Clustering is used which is a hierarchical clustering algorithm. It starts by treating each data
      point as its own cluster and merges them iteratively based on their similarity.
    - Number of clusters can be chosen between 1 - 10.
    - The paraphrase-MiniLM-L6-v2 model from the SentenceTransformer library is used to generate embeddings for the "
      Nature" field in the dataset.
    - These embeddings are dense vector representations of the text, capturing its semantic meaning. Each "Nature" entry
      is transformed into a vector in a high-dimensional space.
    - The algorithm uses cosine distance (computed between the embedding vectors) to measure similarity. Cosine distance
      measures the angle between two vectors, focusing on the direction rather than magnitude, which is well-suited for
      text embeddings.
    - The clusters are formed based on the merging of points with the smallest cosine distance.
    - Since embeddings are in a high-dimensional space, used PCA to reduce them to 2 dimensions for visualization.
    - The reduced 2D representation retains the most important features (variance) of the data, allowing us to plot the
      clusters.
    - Displayed as a scatter plot, with each point representing an incident and the color indicating its cluster.
- **Bar Chart Visualization:** The bar chart provides insights into the frequency distribution of incidents based on
  their "
  Nature." It highlights the most and least common incident types in the dataset.
    - The code uses the Nature column to count occurrences of each unique incident type using
      `df["Nature"].value_counts()`.
    - x-axis represents the unique incident types (Nature).
    - y-axis represents count of incidents for each type.
    - Hovering over the bars reveals exact counts for each incident type.
- **Line Chart Visualization:** The line chart captures temporal patterns in the dataset, showing how incident frequency
  varies across
  different hours of the day.
    - Convert the Date/Time column to a datetime format: `pd.to_datetime(df['Date/Time'])`
    - Extract the hour component: `df['Hour'] = df['Date/Time'].dt.hour`.
    - Group by the hour to calculate the number of incidents in each hour:
      `df.groupby('Hour').size().reset_index(name="Incident Count")`.
    - For consecutive data points, the trend line color indicates the change:
        - Red: Indicates an increase in incident count.
        - Green: Indicates a decrease in incident count.
    - Hovering over the line provides detailed incident counts for specific hours.

### FUNCTIONS

#### `is_header_line(line)`

- Identifies and filters out header lines in the extracted PDF text.

#### `extract_incident_data(file_path)`

- Extracts structured data from PDF files.
- Handles multi-line entries and consolidates them into a single record.

#### `load_data_into_df(data)`

- Loads the extracted data into a Pandas DataFrame with proper column names and data types.

#### `agglomerative_clustering(df)`

- Encodes the "Nature" field using SentenceTransformer.
- Performs clustering using AgglomerativeClustering with cosine distances.
- Reduces the embedding dimensions using PCA for visualization.

### Tests

#### `test_extract_incident_data`

- Mocks the PDF reading process and verifies that data is extracted correctly, checking if the resulting list of data
  matches the expected format.

#### `test_load_data_into_df`

- Verifies that the data is correctly loaded into a pandas DataFrame and the shape and columns are correct.

#### `test_agglomerative_clustering`

- Verifies if the agglomerative clustering function is correctly adding the Cluster column to the DataFrame and if the
  expected number of clusters is generated.

#### `test_valid_url`

- Verifies if the URL validation logic works correctly, adding valid URLs and rejecting invalid ones.

### ASSUMPTIONS:

- **URL**: It is assumed that the URL is correct and PDF file exists.
- **Valid Data Format**: It is assumed that columns will be separated by more than 5 white spaces.
- **Multi Line**: It is assumed that only location part will run into next line and other elements will be part of
  current line.
- **Number of Clusters**: It is assumed that 1-10 clusters will be enough to cluster the data.

### Bugs

- **Clustering Accuracy**: The quality of clustering depends on the SentenceTransformer model and data variety.
- **File submission**: I don't have a threshold on how many files can be handled at a single time. However, Streamlit
  default limit is 200 mb.
- **Sentence Transformer**: The sentence transformer takes some time to load and generate embeddings when multiple files
  are given.
- **Feedback Form**: Backend integration (SMTP or DB) required to store feedback. Currently not supported.