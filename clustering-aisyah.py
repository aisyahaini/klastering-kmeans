#Import Library yang akan digunakan
#%matplotlib inline
import matplotlib.pyplot as plt
import streamlit as st
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from scipy.spatial.distance import cdist
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import silhouette_score, davies_bouldin_score
import seaborn as sns
import base64

#membuat judul web
st.title(""" Aplikasi Klastering K-Means Peningkatan Kualitas Layanan Jasa Pengiriman \n """)

# Fungsi untuk preprocessing data
# Membaca dataset dari file CSV
def preprocessing_data():
    st.header("Prepocessing Data")
    st.write("Disini saya menggunakan dataset dari Perusahaan Jasa Pengiriman Shopee Express dengan jumlah data 1000")
    file_path = 'dataset-spx.csv'
    dataset = pd.read_csv(file_path)

        # Menampilkan dataset
    st.subheader("Tampilan Dataset:")
    st.write(dataset)

        # Pilih hanya kolom kategorikal (gantilah nama kolom sesuai kebutuhan)
    categorical_columns = ['Payment Method', 'Channel', 'Order ID', 'SLS Tracking Number',
                                'Shopee Order SN', 'Sort Code Name', 'SOC Received time',
                                'OnHoldReason', 'Status', 'Order Account', 'Payment Method',
                                'Current Station','Pickup Station', 'Destination Station', 'Next Station']

        # Gunakan Label Encoding untuk setiap kolom kategorikal
    for col in categorical_columns:
        dataset[col+'_encoded'] = dataset[col].astype('category').cat.codes

    # Hapus kolom kategorikal yang sudah diubah
    dataset = dataset.drop(categorical_columns, axis=1)

    # Simpan DataFrame yang sudah diubah ke dalam file CSV
    output_path = 'hasil-num.csv'
    dataset.to_csv(output_path, index=False)
    nama_file = 'hasil-num.csv'

    # Menampilkan dataset setelah preprocessing
    st.subheader("Dataset Setelah Preprocessing:")
    st.write(dataset)

    # Ganti 'nama_file.csv' dengan nama file yang sesuai
    data = pd.read_csv('hasil-num.csv')

    # Pilih semua kolom numerik dari DataFrame
    kolom_numerik = data.select_dtypes(include='number')

    # Hitung matriks korelasi
    matriks_korelasi = kolom_numerik.corr()

    # Buat heatmap menggunakan seaborn
    st.subheader("Heatmap Matriks Korelasi")
    plt.figure(figsize=(10, 8))
    sns.heatmap(matriks_korelasi, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Heatmap Matriks Korelasi')
    st.pyplot(plt)

    # Menghapus kolom tertentu
    kolom_yang_dihapus = ['NO', 'Order ID_encoded', 'SLS Tracking Number_encoded', 'Shopee Order SN_encoded', 
                        'Status_encoded','Payment Method_encoded','SOC Received time_encoded', 
                        'Order Account_encoded','OnHoldReason_encoded', 'Pickup Station_encoded', 
                        'Current Station_encoded','Channel_encoded']
    dataset = dataset.drop(columns=kolom_yang_dihapus)

    # Menampilkan dataset setelah penghapusan kolom
    st.subheader("Dataset Setelah Penghapusan Kolom:")
    st.write(dataset)

    # Menampilkan dataset sebelum penghapusan baris terakhir
    st.subheader("Dataset Sebelum Penghapusan Baris Terakhir:")
    st.write(dataset)

    # Menghapus baris terakhir
    dataset = dataset.drop(dataset.index[-1])

    # Menampilkan dataset setelah penghapusan baris terakhir
    st.subheader("Dataset Setelah Penghapusan Baris Terakhir:")
    st.write(dataset)

    # Menyimpan dataset yang telah diubah ke file CSV baru
    dataset.to_csv('dataset-fix.csv', index=False)


    # Normalisasi data jika diperlukan
    scaler = preprocessing.StandardScaler()
    df_scaled = scaler.fit_transform(dataset)

    # Menampilkan grafik dan tabel hasil normalisasi
    st.subheader("Data Setelah Normalisasi:")
    st.dataframe(pd.DataFrame(df_scaled, columns=dataset.columns))
    
    # Menampilkan histogram dari data setelah normalisasi
    st.subheader("Histogram Data Setelah Normalisasi:")
    for col in dataset.columns:
        plt.figure(figsize=(6, 4))
        plt.hist(df_scaled[:, dataset.columns.get_loc(col)], bins=20, color='teal', alpha=0.7)
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.title(f'Histogram {col} Setelah Normalisasi')
        st.pyplot(plt)
        
# Fungsi untuk visualisasi data
def visualisasi_data():
    st.header("")
    # Load dataset function
def load_dataset(file_path):
    if file_path:
        dataset = pd.read_csv(file_path)
        return dataset
    else:
        st.warning("No file uploaded")
        return None

# Upload CSV file
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

# If a file is uploaded
if uploaded_file is not None:
    # Load the dataset
    dataset = load_dataset(uploaded_file)

    # Display dataset information
    st.write("Kunci Kolom:", dataset.keys())
    st.write("Data Awal:")
    st.write(dataset.head())

    # Convert to array
    X = np.asarray(dataset)

    #==================================== Scatter 2D plot based on user-selected columns ========================================================
    st.subheader("2D Scatter Plot")
    selected_column_x = st.selectbox("Select X-axis column:", dataset.columns)
    selected_column_y = st.selectbox("Select Y-axis column:", dataset.columns)
    fig, ax = plt.subplots()
    ax.scatter(X[:, dataset.columns.get_loc(selected_column_x)], X[:, dataset.columns.get_loc(selected_column_y)], label='True Position')
    ax.set_xlabel(selected_column_x)
    ax.set_ylabel(selected_column_y)
    ax.set_title("Peningkatan Kualitas Layanan Jasa Pengiriman")
    st.pyplot(fig)

    #============================================ 3D Scatter plot Data =============================================================
    st.subheader("Scatter Plot 3D")
    fig_3d = plt.figure(figsize=(16, 8))
    ax_3d = fig_3d.add_subplot(121, projection='3d')  # Scatter plot
    scatter_3d = ax_3d.scatter(
        dataset['Rounding ASF'],
        dataset['Destination Station_encoded'],
        dataset['Sort Code Name_encoded'],
        c=dataset['Rounding ASF'],
        cmap='viridis',
        label='True Position'
    )

    # Menambahkan label pada sumbu scatter plot
    ax_3d.set_xlabel("Rounding ASF")
    ax_3d.set_ylabel("Destination Station_encoded")
    ax_3d.set_zlabel("Sort Code Name_encoded")
    ax_3d.legend()

    # Menampilkan grafik scatter plot 3D
    #st.pyplot(fig_3d)

    # Bar plot untuk 2 kolom terakhir ("Destination", "Next Station")
    ax2 = fig_3d.add_subplot(122)  # Bar plot
    bar_width = 0.35
    bar_positions = range(len(dataset))
    ax2.bar(bar_positions, dataset['Original ASF'], width=bar_width, label='Original ASF')
    ax2.bar([pos + bar_width for pos in bar_positions], dataset['Next Station_encoded'], width=bar_width, label='Next Station_encoded')

    # Menambahkan label dan legenda pada bar plot
    ax2.set_xlabel('Data Points')
    ax2.set_ylabel('Values')
    ax2.legend()

    # Menampilkan grafik bar plot
    plt.title("Peningkatan Kualitas Layanan Jasa Pengiriman")
    plt.show()
    st.pyplot(fig_3d)
    #============================================ 3D Scatter plot Data =============================================================

    # Normalisasi data jika diperlukan
    scaler = preprocessing.StandardScaler()
    df_scaled = scaler.fit_transform(dataset)

    # ===================================== Find optimal k using Elbow method, Silhouette Score, and Davies-Bouldin Index ==================
    # Find optimal k using Elbow method, Silhouette Score, and Davies-Bouldin Index
    st.subheader("Mencari Nilai K Optimal")
    min_k, max_k = 1, 11
    step = 1

    # Distortion plot
    distortions = []
    for n in range(min_k, max_k, step):
        kmeans = KMeans(n_clusters=n, random_state=42)
        kmeans.fit(df_scaled)
        distortions.append(kmeans.inertia_)

    # Silhouette Score plot
    silhouette_scores = []
    for n in range(2, max_k):
        kmeans = KMeans(n_clusters=n, random_state=42)
        labels = kmeans.fit_predict(df_scaled)
        silhouette_avg = silhouette_score(df_scaled, labels)
        silhouette_scores.append(silhouette_avg)

    # Davies-Bouldin Index plot
    db_scores = []
    for k in range(2, max_k):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(df_scaled)
        db_scores.append(davies_bouldin_score(df_scaled, labels))

    # Find optimal k based on the results
    optimal_k_distortion = distortions.index(min(distortions)) + min_k
    optimal_k_silhouette = silhouette_scores.index(max(silhouette_scores)) + 2  # Starting from k=2
    optimal_k_davies = db_scores.index(min(db_scores)) + 2  # Starting from k=2

    # Plotting the results
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 18))

    ax1.plot(range(min_k, max_k, step), distortions, marker='o', color='tab:red', label='Distortion')
    ax1.axvline(x=optimal_k_distortion, color='yellow', linestyle='--', label='Optimal k')
    ax1.set_ylabel('Elbow Method', color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax1.legend(loc='upper left')

    ax2.plot(range(2, max_k), silhouette_scores, marker='o', color='tab:blue', label='Silhouette Score')
    ax2.axvline(x=optimal_k_silhouette, color='red', linestyle='--', label='Optimal k')
    ax2.set_ylabel('Silhouette Score', color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    ax2.legend(loc='upper right')

    ax3.plot(range(2, max_k), db_scores, marker='o', color='tab:green', label='Davies-Bouldin Index')
    ax3.axvline(x=optimal_k_davies, color='orange', linestyle='--', label='Optimal k')
    ax3.set_xlabel('Jumlah Klaster (k)')
    ax3.set_ylabel('Davies-Bouldin Index', color='tab:green')
    ax3.tick_params(axis='y', labelcolor='tab:green')
    ax3.legend(loc='lower right')

    st.pyplot(fig)


    # Tampilkan tabel hasil optimal klaster
    optimal_clusters_data = {
        "Metode": ["Elbow Method", "Silhouette Score", "Davies-Bouldin Index"],
        "Jumlah Optimal Klaster": [optimal_k_distortion, optimal_k_silhouette, optimal_k_davies],
    }
    optimal_clusters_df = pd.DataFrame(optimal_clusters_data)
    st.table(optimal_clusters_df)
    # ===================================== Find optimal k using Elbow method, Silhouette Score, and Davies-Bouldin Index  END =====================

    # ==================================================== Hasil Klastering dengan nilai K =====================================================================
    kmeans = KMeans(n_clusters=optimal_k_silhouette, random_state=42)
    dataset['Cluster'] = kmeans.fit_predict(df_scaled)

    # Display clustering results
    st.subheader("Hasil Klastering:")
    st.write(dataset[['Original ASF', 'Rounding ASF','Destination Station_encoded', 'Next Station_encoded', 'Sort Code Name_encoded', 'Cluster']])

    # Count the number of data points in each cluster
    cluster_counts = dataset['Cluster'].value_counts().sort_index()

    # Create a DataFrame for the cluster counts
    cluster_counts_df = pd.DataFrame({
        'Cluster': cluster_counts.index,
        'Jumlah Data': cluster_counts.values
    })

    # Alternatively, you can also create a bar chart for visual representation
    #st.write("Grafik Jumlah Data dalam Setiap Klaster:")
    fig_cluster_counts = plt.figure()
    plt.bar(cluster_counts.index, cluster_counts.values, color='teal', alpha=0.7)
    plt.xlabel('Klaster')
    plt.ylabel('Jumlah Data')
    plt.title('Jumlah Data dalam Setiap Klaster')
    #st.pyplot(fig_cluster_counts)

    # 2D Scatter plot for visualizing clusters with additional bar plot
    st.subheader("Grafik 2D Hasil Klastering")

    # Memilih kolom untuk sumbu X dan Y
    selected_column_x_2d = st.selectbox("Pilih Kolom untuk Sumbu X:", dataset.columns)
    selected_column_y_2d = st.selectbox("Pilih Kolom untuk Sumbu Y:", dataset.columns)

    # Membuat subplot 2D
    fig_2d_result = plt.figure(figsize=(12, 8))
    ax_2d_result = fig_2d_result.add_subplot(111)  # Scatter plot

    # Scatter plot untuk 2 kolom terpilih
    scatter_2d_result = ax_2d_result.scatter(
        dataset[selected_column_x_2d],
        dataset[selected_column_y_2d],
        c=dataset['Cluster'],
        cmap='viridis',
        label='True Position'
    )

    # Menambahkan label pada sumbu scatter plot
    ax_2d_result.set_xlabel(selected_column_x_2d)
    ax_2d_result.set_ylabel(selected_column_y_2d)
    ax_2d_result.set_title(f"Peningkatan Layanan Jasa Pengiriman - Clustering (k = {optimal_k_silhouette})")
    ax_2d_result.legend()

    # Menampilkan grafik 2D scatter plot dengan klaster
    st.pyplot(fig_2d_result)

    
    # 3D Scatter plot for visualizing clusters with additional bar plot
    st.subheader("Grafik 3D Hasil Klastering")

    # Membuat subplot 3D
    fig_3d_result = plt.figure(figsize=(18, 10))
    ax_3d_result = fig_3d_result.add_subplot(121, projection='3d')  # Scatter plot
    ax_bar_result = fig_3d_result.add_subplot(122)  # Bar plot

    # Scatter plot untuk 3 kolom pertama ("Destination Station_encoded", "Sort Code Name_encoded", "Rounding ASF")
    scatter_3d_result = ax_3d_result.scatter(
        dataset['Destination Station_encoded'],
        dataset['Sort Code Name_encoded'],
        dataset['Rounding ASF'],
        c=dataset['Cluster'],
        cmap='viridis',
        label='True Position'
    )

    # Menambahkan label pada sumbu scatter plot
    ax_3d_result.set_xlabel("Destination Station_encoded")
    ax_3d_result.set_ylabel("Sort Code Name_encoded")
    ax_3d_result.set_zlabel("Rounding ASF")
    ax_3d_result.set_title(f"Peningkatan Layanan Jasa Pengiriman - Clustering (k = {optimal_k_silhouette})")
    ax_3d_result.legend()

    # Bar plot untuk 2 kolom terakhir ("Original ASF", "Next Station_encoded")
    bar_width = 0.35
    bar_positions = np.arange(len(dataset))
    ax_bar_result.bar(bar_positions, dataset['Original ASF'], width=bar_width, color='blue', alpha=0.7, label='Original ASF')
    ax_bar_result.bar(bar_positions + bar_width, dataset['Next Station_encoded'], width=bar_width, color='red', alpha=0.7, label='Next Station_encoded')

    # Menambahkan label dan legenda pada bar plot
    ax_bar_result.set_xlabel('Data Points')
    ax_bar_result.set_ylabel('Values')
    ax_bar_result.legend()

    # Menampilkan grafik 3D scatter plot dengan klaster dan bar plot
    st.pyplot(fig_3d_result)

    # Display the table with cluster counts
    st.write("Jumlah Data dalam Setiap Klaster:")
    st.table(cluster_counts_df)
    
    st.write("Grafik Jumlah Data dalam Setiap Klaster:")
    st.pyplot(fig_cluster_counts)
    # ==================================================== Hasil Klastering dengan nilai K  END=====================================================================


# Fungsi untuk pengujian data
def pengujian_data():
    st.header("Pengujian Data")
    #========================================================== Pengujian Data dgn Silhoutte Index ============================================================
    # Silhouette Index plot
    st.subheader("Pengujian Data Menggunakan Metode Silhouette Index")
    silhouette_scores = []
    for n in range(2, max_k):
        kmeans = KMeans(n_clusters=n, random_state=42)
        labels = kmeans.fit_predict(df_scaled)
        silhouette_avg = silhouette_score(df_scaled, labels)
        silhouette_scores.append(silhouette_avg)

    # Plotting the results including the optimal value
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(range(2, max_k), silhouette_scores, marker='o', color='tab:blue', label='Silhouette Score')
    ax.axvline(x=optimal_k_silhouette, color='red', linestyle='--', label='Optimal k')
    ax.set_xlabel('Number of Clusters (k)')
    ax.set_ylabel('Silhouette Index')
    ax.legend()
    ax.set_title('Silhouette Index for Different Numbers of Clusters')

    # Display the plot
    st.pyplot(fig)
    
    
    # Create a dictionary with the values
    index_values = {
        "Metode": ["Silhouette Index"],
        "Nilai Pengujian": [silhouette_scores[optimal_k_silhouette - 2]]
    }

    # Convert the dictionary to a DataFrame
    index_df = pd.DataFrame(index_values)

    # Display the table
    st.table(index_df)
    
    st.subheader("Kesimpulan")
    st.write("Kesimpulan untuk meng-klastering data jasa pengiriman dengan tujuan peningkatan layanan jasa pengiriman dilakukan dengan 3 cara yaitu memproses data, visualisasi data, dan pengujian data. Dalam memproses data dengan 1000 baris yang data awalnya bertipe kategorikal diubah menjadi numerik dan dinormalisasi. Kemudian dalam visualisasi data berisi mencari nilai K yang optimal menggunakan Sillhoutte Index dan DBI karena dalam kedua metode tersebut mengukur sejauh mana penyebaran objek dalam suatu klaster dibandingkan dengan Elbow Method yang tidak terlalu optimal karena dalam Elbow Method bergantung pada pengamatan objek atau grafik yang bisa saja kurang tepat jika dari pengamatan saja kemudian hasil klastering data beserta grafiknya. Untuk yang terakhir ada pengujian data yang menggunakan Silhoutte Index karena memberikan gambaran tentang seberapa baik pembagian klaster yang telah dilakukan.") 


    #========================================================== Pengujian Data dgn Silhoutte Index END ============================================================

# Main function
def main():
    st.sidebar.title("Aplikasi Klastering K-Means Peningkatan Kualitas Layanan Jasa Pengiriman")
    # Menu sidebar
    menu = ["Preprocessing Data", "Visualisasi Data", "Pengujian Data"]
    choice = st.sidebar.selectbox("Pilih Menu", menu)

    # Tampilkan halaman sesuai dengan pilihan menu
    if choice == "Preprocessing Data":
        preprocessing_data()
    elif choice == "Visualisasi Data":
        visualisasi_data()
    elif choice == "Pengujian Data":
        pengujian_data()

if __name__ == "__main__":
    main()
    
    