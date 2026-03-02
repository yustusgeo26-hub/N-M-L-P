import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# -------------------------------
# Streamlit App Title
# -------------------------------
st.set_page_config(page_title="Student Performance Segmentation", layout="wide")
st.title("📊 Student Performance Segmentation")
st.write("Segment students into High Performer, Average, and At-Risk groups using K-Means clustering.")

# -------------------------------
# Upload Data
# -------------------------------
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])
if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    st.success("File uploaded successfully!")
    st.dataframe(df.head())

    # -------------------------------
    # Select Features
    # -------------------------------
    features = ['ASS1', 'ASS2', 'TEST 1', 'TEST 2']
    if not all(col in df.columns for col in features):
        st.error(f"Excel must contain these columns: {features}")
    else:
        # Convert to numeric & drop invalid rows
        df[features] = df[features].apply(pd.to_numeric, errors='coerce')
        df = df.dropna(subset=features)

        # -------------------------------
        # Scale Features
        # -------------------------------
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[features])

        # -------------------------------
        # Apply K-Means
        # -------------------------------
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        df['Cluster'] = kmeans.fit_predict(X_scaled)

        # -------------------------------
        # Label Clusters
        # -------------------------------
        cluster_means = df.groupby('Cluster')[features].mean()
        cluster_means['Overall'] = cluster_means.mean(axis=1)
        ranking = cluster_means['Overall'].sort_values()
        labels = {
            ranking.index[0]: "At-Risk",
            ranking.index[1]: "Average",
            ranking.index[2]: "High Performer"
        }
        df['Performance_Level'] = df['Cluster'].map(labels)

        st.subheader("Clustered Data")
        st.dataframe(df[['Cluster', 'Performance_Level'] + features].head())

        # -------------------------------
        # Display Cluster Statistics
        # -------------------------------
        st.subheader("Cluster Summary")
        cluster_counts = df['Performance_Level'].value_counts()
        st.bar_chart(cluster_counts)

        # -------------------------------
        # PCA for Visualization
        # -------------------------------
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        fig, ax = plt.subplots(figsize=(8,6))
        scatter = ax.scatter(X_pca[:,0], X_pca[:,1], c=df['Cluster'], cmap='Set1', s=60)
        ax.set_title("Student Performance Clusters (PCA Visualization)")
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.grid(True)

        # Add legend
        handles, _ = scatter.legend_elements()
        ax.legend(handles, ["At-Risk", "Average", "High Performer"], title="Clusters")
        st.pyplot(fig)

        # -------------------------------
        # Optional: Download Results
        # -------------------------------
        st.subheader("Download Clustered Data")
        output = df.copy()
        output_file = "clustered_students.xlsx"
        output.to_excel(output_file, index=False)
        with open(output_file, "rb") as f:
            st.download_button("Download Excel", f, file_name=output_file)
