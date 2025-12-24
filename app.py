import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import warnings
warnings.filterwarnings('ignore')

# =============================================
# KONFIGURASI HALAMAN
# =============================================
st.set_page_config(
    page_title="🎫 IT Ticket Classification",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================
# CUSTOM CSS
# =============================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .prediction-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 1.5rem;
        margin: 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 10px 20px;
        background-color: #f0f2f6;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# =============================================
# FUNGSI LOAD MODEL
# =============================================
@st.cache_resource
def load_lstm_model():
    try:
        model = load_model(
            r"D:\download\Dahsbord_UAP\Model\Model_LSTM\lstm_model.h5"
        )

        with open(
            r"D:\download\Dahsbord_UAP\Model\Model_LSTM\tokenizer.pkl", "rb"
        ) as f:
            tokenizer = pickle.load(f)

        with open(
            r"D:\download\Dahsbord_UAP\Model\Model_LSTM\label_encoder.pkl", "rb"
        ) as f:
            label_encoder = pickle.load(f)

        return model, tokenizer, label_encoder

    except Exception as e:
        st.error(f"Error loading LSTM model: {e}")
        return None, None, None


@st.cache_resource
def load_bert_model():
    try:
        MODEL_PATH = r"D:\download\Dahsbord_UAP\Model\Model_Bert"

        model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
        tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)

        with open(
            r"D:\download\Dahsbord_UAP\Model\Model_Bert\label_encoder.pkl", "rb"
        ) as f:
            label_encoder = pickle.load(f)

        return model, tokenizer, label_encoder

    except Exception as e:
        st.error(f"Error loading BERT model: {e}")
        return None, None, None


@st.cache_resource
def load_distilbert_model():
    try:
        model_path = r'D:\download\Dahsbord_UAP\Model\Model_Distilbert'
        
        # Cek file yang tersedia
        import os
        files = os.listdir(model_path)
        st.info(f"📁 Files in DistilBERT folder: {files}")
        
        # Coba load dengan berbagai cara
        model = None
        
        # Method 1: Load dari safetensors
        try:
            model = DistilBertForSequenceClassification.from_pretrained(
                model_path,
                local_files_only=True,
                use_safetensors=True
            )
            st.success("✅ Model loaded from safetensors")
        except Exception as e1:
            st.warning(f"⚠️ Safetensors failed: {str(e1)[:100]}")
            
            # Method 2: Load dari pytorch_model.bin (jika ada)
            try:
                model = DistilBertForSequenceClassification.from_pretrained(
                    model_path,
                    local_files_only=True,
                    use_safetensors=False
                )
                st.success("✅ Model loaded from pytorch_model.bin")
            except Exception as e2:
                st.warning(f"⚠️ PyTorch model failed: {str(e2)[:100]}")
                
                # Method 3: Load dari HuggingFace Hub (backup)
                try:
                    model = DistilBertForSequenceClassification.from_pretrained(
                        "distilbert-base-uncased",
                        num_labels=10  # Sesuaikan dengan jumlah kelas Anda
                    )
                    st.warning("⚠️ Using default DistilBERT model from HuggingFace")
                except Exception as e3:
                    st.error(f"❌ All methods failed: {str(e3)}")
                    return None, None, None
        
        # Load tokenizer
        tokenizer = DistilBertTokenizer.from_pretrained(
            model_path,
            local_files_only=True
        )

        # Load label encoder
        with open(f"{model_path}\\label_encoder.pkl", "rb") as f:
            label_encoder = pickle.load(f)

        return model, tokenizer, label_encoder

    except Exception as e:
        st.error("❌ DistilBERT gagal dimuat")
        st.error(f"Error: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None, None, None

@st.cache_data
def load_dataset():
    try:
        df = pd.read_csv(
            r'D:\download\Dahsbord_UAP\Dataset\IT Support Ticket Data.csv'
        )

        cols_to_drop = ['Unnamed: 0', 'Tags']
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

        return df

    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

# =============================================
# FUNGSI PREDIKSI
# =============================================
def predict_lstm_distribution(text, model, tokenizer, label_encoder, max_length=350):
    """Prediksi menggunakan LSTM dengan distribusi probabilitas"""
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_length, padding='post')
    pred = model.predict(padded, verbose=0)[0]
    
    labels = label_encoder.classes_
    distribution = {
        labels[i]: float(pred[i])
        for i in range(len(labels))
    }
    
    predicted_label = labels[np.argmax(pred)]
    return predicted_label, distribution

def predict_bert(text, model, tokenizer, label_encoder, max_length=256):
    """Prediksi menggunakan BERT dengan distribusi probabilitas"""
    model.eval()
    inputs = tokenizer(text, return_tensors='pt', truncation=True, 
                       padding=True, max_length=max_length)
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
    
    label = label_encoder.inverse_transform([pred_class])[0]
    
    # Buat distribusi probabilitas untuk semua kelas
    labels = label_encoder.classes_
    probs_array = probs[0].numpy()
    distribution = {
        labels[i]: float(probs_array[i])
        for i in range(len(labels))
    }
    
    return label, distribution

def predict_distilbert(text, model, tokenizer, label_encoder, max_length=256):
    """Prediksi menggunakan DistilBERT dengan distribusi probabilitas, aman terhadap unseen labels"""
    model.eval()
    inputs = tokenizer(text, return_tensors='pt', truncation=True, 
                       padding=True, max_length=max_length)
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
    
    # fallback jika pred_class tidak ada di label_encoder
    labels_list = label_encoder.classes_.tolist()
    if pred_class >= len(labels_list):
        label = "Unknown"
    else:
        label = label_encoder.inverse_transform([pred_class])[0]
    
    # buat distribusi probabilitas
    distribution = {}
    for i in range(probs.shape[1]):
        key = labels_list[i] if i < len(labels_list) else f"Class {i}"
        distribution[key] = float(probs[0][i])
    
    return label, distribution


# =============================================
# SIDEBAR
# =============================================
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/ticket.png", width=150)
    st.markdown("## 🎛️ Pengaturan")
    
    # Pilih Model
    model_choice = st.selectbox(
        "🤖 Pilih Model:",
        ["LSTM", "BERT", "DistilBERT"],
        help="Pilih model machine learning untuk klasifikasi"
    )
    
    st.markdown("---")
    
    # Menu Navigasi
    menu = st.radio(
        "📑 Menu Navigasi:",
        ["🏠 Dashboard", "🔮 Prediksi", "📊 Analisis Data", "📈 Perbandingan Model", "ℹ️ Tentang"]
    )
    
    st.markdown("---")
    st.markdown("### 📌 Info Model")
    
    model_info = {
        "LSTM": "Long Short-Term Memory - Neural network untuk sequence data",
        "BERT": "Bidirectional Encoder Representations from Transformers",
        "DistilBERT": "Versi ringan BERT (menggunakan BodyDepartment)"
    }
    st.info(model_info[model_choice])

# =============================================
# LOAD DATA
# =============================================
df = load_dataset()

# =============================================
# HALAMAN DASHBOARD
# =============================================
if menu == "🏠 Dashboard":
    st.markdown('<h1 class="main-header">🎫 IT Ticket Classification Dashboard</h1>', 
                unsafe_allow_html=True)
    
    if df is not None:
        # Metrics Row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="📝 Total Tiket",
                value=f"{len(df):,}",
                delta="Dataset"
            )
        
        with col2:
            st.metric(
                label="🏢 Jumlah Departemen",
                value=df['Department'].nunique(),
                delta="Kategori"
            )
        
        with col3:
            st.metric(
                label="🔴 Prioritas High",
                value=len(df[df['Priority'] == 'high']),
                delta=f"{len(df[df['Priority'] == 'high'])/len(df)*100:.1f}%"
            )
        
        with col4:
            st.metric(
                label="🤖 Model Aktif",
                value=model_choice,
                delta="Selected"
            )
        
        st.markdown("---")
        
        # Charts Row
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Distribusi Departemen")
            dept_counts = df['Department'].value_counts()
            fig_dept = px.pie(
                values=dept_counts.values,
                names=dept_counts.index,
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_dept.update_traces(textposition='inside', textinfo='percent+label')
            fig_dept.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig_dept, use_container_width=True)
        
        with col2:
            st.markdown("### 📈 Distribusi Prioritas")
            priority_counts = df['Priority'].value_counts()
            colors = {'high': '#FF6B6B', 'medium': '#FFE66D', 'low': '#4ECDC4'}
            fig_priority = px.bar(
                x=priority_counts.index,
                y=priority_counts.values,
                color=priority_counts.index,
                color_discrete_map=colors
            )
            fig_priority.update_layout(
                xaxis_title="Prioritas",
                yaxis_title="Jumlah Tiket",
                showlegend=False,
                height=400
            )
            st.plotly_chart(fig_priority, use_container_width=True)
        
        # Heatmap Departemen vs Prioritas
        st.markdown("### 🗺️ Heatmap: Departemen vs Prioritas")
        pivot_table = pd.crosstab(df['Department'], df['Priority'])
        fig_heatmap = px.imshow(
            pivot_table,
            labels=dict(x="Prioritas", y="Departemen", color="Jumlah"),
            color_continuous_scale='Blues',
            aspect='auto'
        )
        fig_heatmap.update_layout(height=400)
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Sample Data
        st.markdown("### 📋 Sample Data Tiket")
        st.dataframe(
            df[['Body', 'Department', 'Priority']].head(10),
            use_container_width=True,
            height=300
        )

# =============================================
# HALAMAN PREDIKSI
# =============================================
elif menu == "🔮 Prediksi":
    st.markdown('<h1 class="main-header">🔮 Prediksi Departemen Tiket</h1>', 
                unsafe_allow_html=True)
    
    st.markdown(f"**Model yang digunakan:** `{model_choice}`")
    
    # Input Text
    st.markdown("### ✍️ Masukkan Teks Tiket")
    
    # Sample texts untuk demo
    sample_texts = {
        "Pilih contoh...": "",
        "Technical Support": "Dear Customer Support Team, I am writing to report a critical system outage affecting our IT infrastructure. The server has been down for several hours and we need immediate assistance.",
        "Billing and Payments": "Dear Support Team, I hope this message finds you well. I have a question regarding my recent invoice. There seems to be a discrepancy in the billing amount.",
        "Returns and Exchanges": "Dear Customer Support, I would like to request a return for a product I recently purchased. The item arrived damaged and I need a replacement.",
        "Sales and Pre-Sales": "Hello Team, I am interested in learning more about your enterprise solutions. Could you please provide pricing information and a product demo?"
    }
    
    selected_sample = st.selectbox("📌 Pilih contoh teks (opsional):", list(sample_texts.keys()))
    
    input_text = st.text_area(
        "Masukkan teks tiket di sini:",
        value=sample_texts[selected_sample],
        height=200,
        placeholder="Ketik atau paste teks tiket layanan IT Anda di sini..."
    )
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        predict_btn = st.button("🚀 Prediksi Sekarang", use_container_width=True, type="primary")
    
    if predict_btn and input_text:
        with st.spinner(f"🔄 Memproses dengan {model_choice}..."):
            try:
                distribution = None
                label = None
                
                # Load model sesuai pilihan
                if model_choice == "LSTM":
                    model, tokenizer, label_encoder = load_lstm_model()
                    if model:
                        label, distribution = predict_lstm_distribution(
                            input_text, model, tokenizer, label_encoder
                        )
                
                elif model_choice == "BERT":
                    model, tokenizer, label_encoder = load_bert_model()
                    if model:
                        label, distribution = predict_bert(
                            input_text, model, tokenizer, label_encoder
                        )
                
                else:  # DistilBERT
                    model, tokenizer, label_encoder = load_distilbert_model()
                    if model:
                        label, distribution = predict_distilbert(
                            input_text, model, tokenizer, label_encoder
                        )
                
                # Tampilkan hasil
                if label and distribution:
                    st.success("✅ Prediksi Berhasil!")
                    
                    # Tampilkan prediksi departemen
                    st.markdown(f"""
                    <div class="prediction-box">
                        <h3>🏢 Departemen Prediksi</h3>
                        <h2>{label}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # DISTRIBUSI PROBABILITAS
                    st.markdown("### 📊 Distribusi Probabilitas per Departemen")
                    
                    prob_df = pd.DataFrame({
                        "Departemen": list(distribution.keys()),
                        "Probabilitas": [v * 100 for v in distribution.values()]
                    }).sort_values("Probabilitas", ascending=True)
                    
                    fig_prob = px.bar(
                        prob_df,
                        x="Probabilitas",
                        y="Departemen",
                        orientation="h",
                        color="Probabilitas",
                        color_continuous_scale="Viridis",
                        text="Probabilitas"
                    )
                    fig_prob.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
                    fig_prob.update_layout(
                        xaxis_title="Probabilitas (%)",
                        yaxis_title="",
                        height=350,
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig_prob, use_container_width=True)
                    
            except Exception as e:
                st.error(f"❌ Error saat prediksi: {e}")
                import traceback
                st.error(traceback.format_exc())
    
    elif predict_btn and not input_text:
        st.warning("⚠️ Mohon masukkan teks tiket terlebih dahulu!")

# =============================================
# HALAMAN ANALISIS DATA
# =============================================
elif menu == "📊 Analisis Data":
    st.markdown('<h1 class="main-header">📊 Analisis Data Tiket</h1>', 
                unsafe_allow_html=True)
    
    if df is not None:
        tab1, tab2, tab3 = st.tabs(["📈 Statistik", "📝 Teks Analisis", "🔍 Eksplorasi Data"])
        
        with tab1:
            st.markdown("### 📌 Statistik Deskriptif")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Distribusi per Departemen")
                dept_stats = df['Department'].value_counts().reset_index()
                dept_stats.columns = ['Departemen', 'Jumlah']
                dept_stats['Persentase'] = (dept_stats['Jumlah'] / len(df) * 100).round(2)
                st.dataframe(dept_stats, use_container_width=True)
            
            with col2:
                st.markdown("#### Distribusi per Prioritas")
                priority_stats = df['Priority'].value_counts().reset_index()
                priority_stats.columns = ['Prioritas', 'Jumlah']
                priority_stats['Persentase'] = (priority_stats['Jumlah'] / len(df) * 100).round(2)
                st.dataframe(priority_stats, use_container_width=True)
            
            # Stacked Bar Chart
            st.markdown("### 📊 Departemen berdasarkan Prioritas")
            stacked_data = df.groupby(['Department', 'Priority']).size().unstack(fill_value=0)
            fig_stacked = px.bar(
                stacked_data,
                barmode='stack',
                color_discrete_map={'high': '#FF6B6B', 'medium': '#FFE66D', 'low': '#4ECDC4'}
            )
            fig_stacked.update_layout(
                xaxis_title="Departemen",
                yaxis_title="Jumlah Tiket",
                legend_title="Prioritas"
            )
            st.plotly_chart(fig_stacked, use_container_width=True)
        
        with tab2:
            st.markdown("### 📝 Analisis Panjang Teks")
            
            # Hitung panjang teks
            df['text_length'] = df['Body'].apply(lambda x: len(str(x)))
            df['word_count'] = df['Body'].apply(lambda x: len(str(x).split()))
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Rata-rata Karakter", f"{df['text_length'].mean():.0f}")
            with col2:
                st.metric("Rata-rata Kata", f"{df['word_count'].mean():.0f}")
            with col3:
                st.metric("Max Kata", f"{df['word_count'].max():.0f}")
            
            # Histogram Panjang Teks
            fig_hist = px.histogram(
                df, x='word_count',
                nbins=50,
                color='Department',
                title="Distribusi Jumlah Kata per Tiket"
            )
            st.plotly_chart(fig_hist, use_container_width=True)
            
            # Box Plot
            fig_box = px.box(
                df, x='Department', y='word_count',
                color='Priority',
                title="Distribusi Jumlah Kata per Departemen dan Prioritas"
            )
            st.plotly_chart(fig_box, use_container_width=True)
        
        with tab3:
            st.markdown("### 🔍 Eksplorasi Data")
            
            # Filter
            col1, col2 = st.columns(2)
            with col1:
                selected_dept = st.multiselect(
                    "Filter Departemen:",
                    options=df['Department'].unique(),
                    default=df['Department'].unique()
                )
            with col2:
                selected_priority = st.multiselect(
                    "Filter Prioritas:",
                    options=df['Priority'].unique(),
                    default=df['Priority'].unique()
                )
            
            filtered_df = df[
                (df['Department'].isin(selected_dept)) & 
                (df['Priority'].isin(selected_priority))
            ]
            
            st.markdown(f"**Menampilkan {len(filtered_df)} dari {len(df)} data**")
            st.dataframe(
                filtered_df[['Body', 'Department', 'Priority']],
                use_container_width=True,
                height=400
            )

# =============================================
# HALAMAN PERBANDINGAN MODEL
# =============================================
elif menu == "📈 Perbandingan Model":
    st.markdown('<h1 class="main-header">📈 Perbandingan Model</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("### 🏆 Performa Model")
    
    # Data performa model berdasarkan hasil aktual
    performance_data = {
        'Model': ['LSTM', 'BERT', 'DistilBERT'],
        'Accuracy': [0.64, 0.82, 0.67],
        'Precision': [0.65, 0.83, 0.68],
        'Recall': [0.64, 0.82, 0.67],
        'F1-Score': [0.64, 0.83, 0.66],
        'Training Time (min)': [15, 45, 30],
        'Inference Time (ms)': [10, 50, 25]
    }
    
    perf_df = pd.DataFrame(performance_data)
    
    # Tabel Performa
    st.dataframe(perf_df, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Radar Chart
        st.markdown("### 📡 Radar Chart - Metrik Evaluasi")
        
        categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        fig_radar = go.Figure()
        
        for i, model in enumerate(['LSTM', 'BERT', 'DistilBERT']):
            values = perf_df[perf_df['Model'] == model][categories].values.flatten().tolist()
            values += values[:1]  # Close the loop
            
            fig_radar.add_trace(go.Scatterpolar(
                r=values,
                theta=categories + [categories[0]],
                fill='toself',
                name=model
            ))
        
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            height=400
        )
        st.plotly_chart(fig_radar, use_container_width=True)
    
    with col2:
        # Bar Chart Perbandingan
        st.markdown("### 📊 Perbandingan Akurasi")
        
        fig_bar = px.bar(
            perf_df,
            x='Model',
            y='Accuracy',
            color='Model',
            text='Accuracy',
            color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1']
        )
        fig_bar.update_traces(texttemplate='%{text:.2%}', textposition='outside')
        fig_bar.update_layout(
            yaxis_range=[0, 1],
            showlegend=False,
            height=400
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Time Comparison
    st.markdown("### ⏱️ Perbandingan Waktu")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_train = px.bar(
            perf_df,
            x='Model',
            y='Training Time (min)',
            color='Model',
            title="Waktu Training",
            color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1']
        )
        st.plotly_chart(fig_train, use_container_width=True)
    
    with col2:
        fig_infer = px.bar(
            perf_df,
            x='Model',
            y='Inference Time (ms)',
            color='Model',
            title="Waktu Inferensi",
            color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1']
        )
        st.plotly_chart(fig_infer, use_container_width=True)
    
    # Rekomendasi
    st.markdown("### 💡 Rekomendasi")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("""
        **🔵 LSTM**
        - ✅ Ringan & Cepat
        - ✅ Resource minimal
        - ⚠️ Akurasi lebih rendah
        - 📌 Cocok untuk deployment ringan
        """)
    
    with col2:
        st.success("""
        **🟢 BERT**
        - ✅ Akurasi tertinggi
        - ✅ Pemahaman konteks baik
        - ⚠️ Resource tinggi
        - 📌 Cocok untuk akurasi maksimal
        """)
    
    with col3:
        st.warning("""
        **🟡 DistilBERT**
        - ✅ Balance akurasi & kecepatan
        - ✅ Menggunakan BodyDepartment
        - ✅ Lebih ringan dari BERT
        - 📌 Pilihan terbaik secara umum
        """)

# =============================================
# HALAMAN TENTANG
# =============================================
elif menu == "ℹ️ Tentang":
    st.markdown('<h1 class="main-header">ℹ️ Tentang Aplikasi</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    ## 🎫 IT Ticket Classification Dashboard
    
    Dashboard ini digunakan untuk mengklasifikasikan tiket layanan IT ke departemen yang tepat
    menggunakan berbagai model Machine Learning dan Deep Learning.
    
    ### 🎯 Fitur Utama
    
    | Fitur | Deskripsi |
    |-------|-----------|
    | 🏠 Dashboard | Visualisasi statistik dataset |
    | 🔮 Prediksi | Klasifikasi tiket real-time |
    | 📊 Analisis Data | Eksplorasi dan analisis data |
    | 📈 Perbandingan Model | Komparasi performa model |
    
    ### 🤖 Model yang Tersedia
    
    1. **LSTM** (Long Short-Term Memory)
       - Neural network untuk sequence data
       - Cepat dan ringan
    
    2. **BERT** (Bidirectional Encoder Representations from Transformers)
       - State-of-the-art NLP model
       - Akurasi tertinggi
    
    3. **DistilBERT**
       - Versi ringan dari BERT
       - Menggunakan fitur **BodyDepartment**
       - Balance antara akurasi dan kecepatan
    
    ### 🏢 Kategori Departemen
    
    - Technical Support
    - Billing and Payments
    - Returns and Exchanges
    - Sales and Pre-Sales
    
    ### 📊 Prioritas Tiket
    
    - 🔴 **High** - Urgent, memerlukan penanganan segera
    - 🟡 **Medium** - Prioritas sedang
    - 🟢 **Low** - Dapat ditangani sesuai jadwal
    
    ---
    
    ### 👨‍💻 Teknologi yang Digunakan
    
    - **Streamlit** - Dashboard Framework
    - **TensorFlow/Keras** - LSTM Model
    - **Transformers (HuggingFace)** - BERT & DistilBERT
    - **Plotly** - Visualisasi Interaktif
    - **Pandas** - Data Processing
    
    """)
    
    st.markdown("---")
    st.markdown("© 2025 IT Ticket Classification Dashboard | Created with ❤️ using Streamlit")

# =============================================
# FOOTER
# =============================================
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <p>🎫 IT Ticket Classification Dashboard v1.0 | Model: """ + model_choice + """</p>
    </div>
    """,
    unsafe_allow_html=True
)