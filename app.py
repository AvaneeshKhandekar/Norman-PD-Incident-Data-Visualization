import streamlit as st
import pandas as pd
import plotly.express as px
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from pypdf import PdfReader
import os
import re
import urllib.request
import tempfile
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_distances


def is_header_line(line):
    header_keywords = {
        "NORMAN POLICE DEPARTMENT",
        "Daily Incident Summary (Public)",
        "Location",
        "Date/ Time",
        "Nature",
        "Incident ORI"
    }
    return any(keyword in line for keyword in header_keywords)


def extract_incident_data(file_path):
    try:
        pdf_reader = PdfReader(file_path)
        extracted_data = []
        for page in pdf_reader.pages:
            text = page.extract_text(extraction_mode="layout")
            if text is None:
                continue
            for line in text.split("\n"):
                if is_header_line(line):
                    continue
                attributes = re.split(r"[ \t\r\n]{5,}", line.strip())
                n = len(attributes)
                if n == 5:
                    extracted_data.append(attributes)
                elif n < 5:
                    if extracted_data:
                        extracted_data[-1][2] += ' '.join(attributes)
        return extracted_data
    except:
        return None


def load_data_into_df(data):
    columns = ["Date/Time", "Incident Number", "Location", "Nature", "Incident ORI"]
    dtypes = {
        "Date/Time": "string",
        "Incident Number": "string",
        "Location": "string",
        "Nature": "string",
        "Incident ORI": "string",
    }
    df = pd.DataFrame(data, columns=columns).astype(dtypes)
    return df


def agglomerative_clustering(df, n_clusters):
    distance_threshold = None
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    embeddings = model.encode(df['Nature'].tolist(), batch_size=64, show_progress_bar=True, convert_to_tensor=True)
    cosine_dist_matrix = cosine_distances(embeddings)
    if n_clusters == 0:
        n_clusters = None
        distance_threshold = 0.5
    clustering_model = AgglomerativeClustering(
        metric='precomputed',
        linkage='average',
        n_clusters=n_clusters,
        distance_threshold=distance_threshold,
    )
    cluster_labels = clustering_model.fit_predict(cosine_dist_matrix)

    df['Cluster'] = cluster_labels

    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)

    df['PCA_X'] = reduced_embeddings[:, 0]
    df['PCA_Y'] = reduced_embeddings[:, 1]

    return df


st.title("Norman Police Incident Reports")

uploaded_files = st.file_uploader("Upload NormanPD Incident Summary PDFs", type="pdf", accept_multiple_files=True)
submitted_urls = st.text_area("Paste URLs of NormanPD Incident Summary PDFs (comma-separated):")
valid_urls = []

if submitted_urls:
    urls = [url.strip() for url in submitted_urls.split(",")]
    for url in urls:
        try:
            link = re.compile(
                '(?i)((?:https?://|www\d{0,3}[.])?[a-z0-9.\-]+[.](?:(?:international)|(?:construction)|(?:contractors)|(?:enterprises)|(?:photography)|(?:immobilien)|(?:management)|(?:technology)|(?:directory)|(?:education)|(?:equipment)|(?:institute)|(?:marketing)|(?:solutions)|(?:builders)|(?:clothing)|(?:computer)|(?:democrat)|(?:diamonds)|(?:graphics)|(?:holdings)|(?:lighting)|(?:plumbing)|(?:training)|(?:ventures)|(?:academy)|(?:careers)|(?:company)|(?:domains)|(?:florist)|(?:gallery)|(?:guitars)|(?:holiday)|(?:kitchen)|(?:recipes)|(?:shiksha)|(?:singles)|(?:support)|(?:systems)|(?:agency)|(?:berlin)|(?:camera)|(?:center)|(?:coffee)|(?:estate)|(?:kaufen)|(?:luxury)|(?:monash)|(?:museum)|(?:photos)|(?:repair)|(?:social)|(?:tattoo)|(?:travel)|(?:viajes)|(?:voyage)|(?:build)|(?:cheap)|(?:codes)|(?:dance)|(?:email)|(?:glass)|(?:house)|(?:ninja)|(?:photo)|(?:shoes)|(?:solar)|(?:today)|(?:aero)|(?:arpa)|(?:asia)|(?:bike)|(?:buzz)|(?:camp)|(?:club)|(?:coop)|(?:farm)|(?:gift)|(?:guru)|(?:info)|(?:jobs)|(?:kiwi)|(?:land)|(?:limo)|(?:link)|(?:menu)|(?:mobi)|(?:moda)|(?:name)|(?:pics)|(?:pink)|(?:post)|(?:rich)|(?:ruhr)|(?:sexy)|(?:tips)|(?:wang)|(?:wien)|(?:zone)|(?:biz)|(?:cab)|(?:cat)|(?:ceo)|(?:com)|(?:edu)|(?:gov)|(?:int)|(?:mil)|(?:net)|(?:onl)|(?:org)|(?:pro)|(?:red)|(?:tel)|(?:uno)|(?:xxx)|(?:ac)|(?:ad)|(?:ae)|(?:af)|(?:ag)|(?:ai)|(?:al)|(?:am)|(?:an)|(?:ao)|(?:aq)|(?:ar)|(?:as)|(?:at)|(?:au)|(?:aw)|(?:ax)|(?:az)|(?:ba)|(?:bb)|(?:bd)|(?:be)|(?:bf)|(?:bg)|(?:bh)|(?:bi)|(?:bj)|(?:bm)|(?:bn)|(?:bo)|(?:br)|(?:bs)|(?:bt)|(?:bv)|(?:bw)|(?:by)|(?:bz)|(?:ca)|(?:cc)|(?:cd)|(?:cf)|(?:cg)|(?:ch)|(?:ci)|(?:ck)|(?:cl)|(?:cm)|(?:cn)|(?:co)|(?:cr)|(?:cu)|(?:cv)|(?:cw)|(?:cx)|(?:cy)|(?:cz)|(?:de)|(?:dj)|(?:dk)|(?:dm)|(?:do)|(?:dz)|(?:ec)|(?:ee)|(?:eg)|(?:er)|(?:es)|(?:et)|(?:eu)|(?:fi)|(?:fj)|(?:fk)|(?:fm)|(?:fo)|(?:fr)|(?:ga)|(?:gb)|(?:gd)|(?:ge)|(?:gf)|(?:gg)|(?:gh)|(?:gi)|(?:gl)|(?:gm)|(?:gn)|(?:gp)|(?:gq)|(?:gr)|(?:gs)|(?:gt)|(?:gu)|(?:gw)|(?:gy)|(?:hk)|(?:hm)|(?:hn)|(?:hr)|(?:ht)|(?:hu)|(?:id)|(?:ie)|(?:il)|(?:im)|(?:in)|(?:io)|(?:iq)|(?:ir)|(?:is)|(?:it)|(?:je)|(?:jm)|(?:jo)|(?:jp)|(?:ke)|(?:kg)|(?:kh)|(?:ki)|(?:km)|(?:kn)|(?:kp)|(?:kr)|(?:kw)|(?:ky)|(?:kz)|(?:la)|(?:lb)|(?:lc)|(?:li)|(?:lk)|(?:lr)|(?:ls)|(?:lt)|(?:lu)|(?:lv)|(?:ly)|(?:ma)|(?:mc)|(?:md)|(?:me)|(?:mg)|(?:mh)|(?:mk)|(?:ml)|(?:mm)|(?:mn)|(?:mo)|(?:mp)|(?:mq)|(?:mr)|(?:ms)|(?:mt)|(?:mu)|(?:mv)|(?:mw)|(?:mx)|(?:my)|(?:mz)|(?:na)|(?:nc)|(?:ne)|(?:nf)|(?:ng)|(?:ni)|(?:nl)|(?:no)|(?:np)|(?:nr)|(?:nu)|(?:nz)|(?:om)|(?:pa)|(?:pe)|(?:pf)|(?:pg)|(?:ph)|(?:pk)|(?:pl)|(?:pm)|(?:pn)|(?:pr)|(?:ps)|(?:pt)|(?:pw)|(?:py)|(?:qa)|(?:re)|(?:ro)|(?:rs)|(?:ru)|(?:rw)|(?:sa)|(?:sb)|(?:sc)|(?:sd)|(?:se)|(?:sg)|(?:sh)|(?:si)|(?:sj)|(?:sk)|(?:sl)|(?:sm)|(?:sn)|(?:so)|(?:sr)|(?:st)|(?:su)|(?:sv)|(?:sx)|(?:sy)|(?:sz)|(?:tc)|(?:td)|(?:tf)|(?:tg)|(?:th)|(?:tj)|(?:tk)|(?:tl)|(?:tm)|(?:tn)|(?:to)|(?:tp)|(?:tr)|(?:tt)|(?:tv)|(?:tw)|(?:tz)|(?:ua)|(?:ug)|(?:uk)|(?:us)|(?:uy)|(?:uz)|(?:va)|(?:vc)|(?:ve)|(?:vg)|(?:vi)|(?:vn)|(?:vu)|(?:wf)|(?:ws)|(?:ye)|(?:yt)|(?:za)|(?:zm)|(?:zw))(?:/[^\s()<>]+[^\s`!()\[\]{};:\'".,<>?\xab\xbb\u201c\u201d\u2018\u2019])?)',
                re.IGNORECASE)
            if not url or not re.match(link, url):
                st.error(f"Invalid URL: {url}")
            else:
                valid_urls.append(url)
        except:
            st.error(f"Invalid URL: {url}")

if uploaded_files or valid_urls:
    data = []
    if uploaded_files:
        for file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(file.read())
                extracted_data = extract_incident_data(tmp_file.name)
                if extracted_data:
                    data.extend(extracted_data)
                else:
                    st.error(f"Invalid data uploaded in {file.name}. Could not extract incident information.")

    if valid_urls:
        for url in valid_urls:
            file_name = os.path.basename(url)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                try:
                    urllib.request.urlretrieve(url, tmp_file.name)
                    extracted_data = extract_incident_data(tmp_file.name)
                    if extracted_data:
                        data.extend(extracted_data)
                    else:
                        st.error(f"Invalid data extracted from URL: {url}")
                except Exception as e:
                    st.error(f"Failed to retrieve or process URL {url}: {e}")

    if data:
        df = load_data_into_df(data)

        st.subheader("Raw Data")
        st.dataframe(df)

        st.subheader("Agglomerative Clustering Visualization")
        st.write("""
            This visualization shows the results of Agglomerative Clustering on the 'Nature' of incidents.
            The Agglomerative Clustering algorithm groups incidents based on their semantic similarity in their 'Nature'.
            Each cluster represents a group of incidents with similar characteristics. The high-dimensional embeddings of the 
            'Nature' text are reduced to two dimensions using PCA, to display the clusters in a 2D space. 
            Each point in the scatter plot represents an incident, with the color indicating its assigned cluster. 
            This visualization helps in understanding which nature of incidents are similar.
        """)
        if not df.empty:
            num_clusters = st.slider("Select Number of Clusters", min_value=1, max_value=10, value=0, step=1)

            df = agglomerative_clustering(df, num_clusters)

            fig = px.scatter(df, x='PCA_X', y='PCA_Y', color='Cluster',
                             color_continuous_scale='viridis',
                             title="Agglomerative Clustering of Incidents Based on Semantic Similarity",
                             hover_data=['Nature'])

            fig.update_layout(
                xaxis_title="PCA Component 1",
                yaxis_title="PCA Component 2",
                autosize=True,
                width=1500,
                height=700
            )

            st.plotly_chart(fig)

            st.subheader("Nature of Incidents")
            st.write("""
                This bar chart displays the frequency of each type of incident ('Nature'). 
                It provides an overview of the most common incident types in the dataset. 
                Hover over the bars to see the exact count for each incident type.
            """)
            nature_counts = df["Nature"].value_counts()

            fig = px.bar(
                nature_counts,
                x=nature_counts.index,
                y=nature_counts.values,
                labels={'x': 'Nature', 'y': 'Count'},
                title="Count of Incidents by Nature",
                color=nature_counts.values,
                color_continuous_scale='mint'
            )

            fig.update_layout(
                autosize=True,
                width=1500,
                height=700,
                title_x=0.5,
                xaxis_tickangle=-45
            )

            st.plotly_chart(fig)

            st.subheader("Incident Trend Over Time")
            st.write("""
                This line chart shows the number of incidents that occurred during each hour of the day.
                It helps to identify patterns or peak times when incidents are most frequent.
                Hover over the line to see the number of incidents at each hour.
            """)

            df['Date/Time'] = pd.to_datetime(df['Date/Time'])
            df['Hour'] = df['Date/Time'].dt.hour

            hourly_trend = df.groupby('Hour').size().reset_index(name="Incident Count")

            fig = go.Figure()

            for i in range(1, len(hourly_trend)):
                x_values = [hourly_trend['Hour'][i - 1], hourly_trend['Hour'][i]]
                y_values = [hourly_trend['Incident Count'][i - 1], hourly_trend['Incident Count'][i]]

                if y_values[1] > y_values[0]:
                    line_color = 'red'
                else:
                    line_color = 'lightgreen'

                fig.add_trace(go.Scatter(
                    x=x_values,
                    y=y_values,
                    mode='lines',
                    line=dict(color=line_color, width=3)
                ))

            fig.update_layout(
                title="Number of Incidents by Hour",
                xaxis_title="Hour of Day",
                yaxis_title="Number of Incidents",
                autosize=True,
                width=1500,
                height=700,
            )

            st.plotly_chart(fig)

    st.subheader("Provide Feedback")
    feedback = st.text_area("I would love to hear your feedback or suggestions about this tool.")
    if st.button("Submit Feedback"):
        if feedback:
            st.success("Thank you for your feedback!")
        else:
            st.warning("Please provide some feedback before submitting.")
