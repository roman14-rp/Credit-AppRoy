import joblib
import pandas as pd
import streamlit as st

from fitur_kredit import buat_fitur

st.set_page_config(page_title="Cek Kelayakan Pinjaman", page_icon="💳", layout="centered")

st.markdown(
    """
<style>
.hasil {border-radius: 10px; padding: 16px 20px; margin-top: 16px; border: 1px solid;}
.hasil.ok {background: #0f2a14; border-color: #14532d;}
.hasil.ok .judul {color: #16a34a;}
.hasil.ok .isi {background: #16a34a;}
.hasil.no {background: #2a0f0f; border-color: #7f1d1d;}
.hasil.no .judul {color: #ef4444;}
.hasil.no .isi {background: #ef4444;}
.hasil .judul {font-size: 1.15rem; font-weight: 600; margin-bottom: 10px;}
.hasil .bar {height: 8px; background: #262626; border-radius: 4px; overflow: hidden; margin-bottom: 10px;}
.hasil .isi {height: 100%;}
div[data-testid="stFormSubmitButton"] button {width: 100%;}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def muat_model():
    return joblib.load("model_kredit.joblib")


model = muat_model()

TENOR = [12, 36, 60, 84, 120, 180, 240, 300, 360, 480]
RIWAYAT = {"Baik": 1.0, "Buruk": 0.0}
MENIKAH = {"Ya": "Yes", "Tidak": "No"}
PENDIDIKAN = {"Sarjana (Graduate)": "Graduate", "Non-sarjana": "Not Graduate"}
LOKASI = {"Perkotaan": "Urban", "Pinggiran kota": "Semiurban", "Pedesaan": "Rural"}
AMBANG = 0.5  # peluang >= 0.5 dianggap disetujui

with st.form("form_kredit"):
    st.markdown("## Cek kelayakan pinjaman")
    st.caption("Isi data di bawah, lalu klik Cek kelayakan.")

    kiri, kanan = st.columns(2)
    with kiri:
        pemohon = st.number_input("Pendapatan pemohon", min_value=0, value=5000, step=100)
        pendamping = st.number_input("Pendapatan pendamping", min_value=0, value=0, step=100)
        jumlah = st.number_input("Jumlah pinjaman (ribuan)", min_value=1, value=130, step=10)
    with kanan:
        riwayat = st.selectbox("Riwayat kredit", list(RIWAYAT))
        tenor = st.selectbox("Tenor pinjaman (bulan)", TENOR, index=TENOR.index(360))

    with st.expander("Data tambahan (opsional): status menikah, pendidikan, lokasi properti"):
        menikah = st.selectbox("Status menikah", list(MENIKAH))
        pendidikan = st.selectbox("Pendidikan", list(PENDIDIKAN))
        lokasi = st.selectbox("Lokasi properti", list(LOKASI), index=1)

    kirim = st.form_submit_button("Cek kelayakan")

if kirim:
    data = pd.DataFrame([{
        "ApplicantIncome": pemohon,
        "CoapplicantIncome": pendamping,
        "LoanAmount": jumlah,
        "Loan_Amount_Term": tenor,
        "Credit_History": RIWAYAT[riwayat],
        "Married": MENIKAH[menikah],
        "Education": PENDIDIKAN[pendidikan],
        "Property_Area": LOKASI[lokasi],
    }])
    peluang = float(model.predict_proba(buat_fitur(data))[0][1])
    setuju = peluang >= AMBANG
    kelas = "ok" if setuju else "no"
    judul = "Pinjaman diprediksi disetujui" if setuju else "Pinjaman diprediksi tidak disetujui"
    persen = round(peluang * 100)

    st.markdown(
        f"""
<div class="hasil {kelas}">
<div class="judul">{judul}</div>
<div class="bar"><div class="isi" style="width:{persen}%"></div></div>
<div>Peluang disetujui: {persen}%</div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.caption("Ini perkiraan dari model machine learning, bukan keputusan resmi pemberian kredit.")
