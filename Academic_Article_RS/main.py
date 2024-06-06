from pymongo import MongoClient
import torch
import numpy as np
from transformers import BertModel, BertTokenizer
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, redirect, url_for, render_template, request
import fasttext
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from datasets import load_dataset

# MongoDB bağlantısı
mongoclient = MongoClient("mongodb://localhost:27017")
db = mongoclient["metin_benzerlik_db"]
makaleler = db["makaleler"]
kullanicilar = db["kullanicilar"]
ilgiler = db["ilgiler"]

# Modelleri ve durak kelimeleri yükleme

print("Modeller yükleniyor...")
fasttext_model = fasttext.load_model("C:/Users/Mehmet Yilmaz/Desktop/cc.en.300.bin")
stopwords = set(stopwords.words('english'))
stemmer = PorterStemmer()
scibert_t = BertTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
scibert_m = BertModel.from_pretrained("allenai/scibert_scivocab_uncased")


# Metni FastText vektörüne dönüştürme
def metin_to_ft_vektor(metin):
    filtrelenmis_tokenler = []
    for kelime in word_tokenize(metin):
        if kelime not in stopwords:
            filtrelenmis_tokenler.append(kelime)
    kok_tokenler = []
    for kelime in filtrelenmis_tokenler:
        kok_tokenler.append(stemmer.stem(kelime))
    kelime_vektorleri = []
    for kelime in kok_tokenler:
        kelime_vektorleri.append(fasttext_model.get_word_vector(kelime))
    return np.mean(kelime_vektorleri, axis=0)


# Metni SciBERT vektörüne dönüştürme
def metin_to_bert_vektor(metin):
    filtrelenmis_tokenler = []
    for kelime in word_tokenize(metin):
        if kelime not in stopwords:
            filtrelenmis_tokenler.append(kelime)
    kok_tokenler = []
    for kelime in filtrelenmis_tokenler:
        kok_tokenler.append(stemmer.stem(kelime))
    girdiler = scibert_t(" ".join(kok_tokenler), return_tensors="pt")
    with torch.no_grad():
        ciktilar = scibert_m(**girdiler)
    return torch.mean(ciktilar.last_hidden_state, dim=1)

#Veri seti aktarma inspec

if makaleler.estimated_document_count() == 0:
    print("Veri seti veri tabanına aktarılıyor")
    nltk.download('punkt')
    nltk.download('stopwords')
    dataset = load_dataset("midas/inspec", "raw")

    i = 0
    for split in ["train", "validation", "test"]:
        for d in dataset[split]:
            i += 1
            try:
                doc_text = " ".join(d["document"])
                doc_ft_vector = metin_to_ft_vektor(doc_text).tolist()
                doc_bert_vector = metin_to_bert_vektor(doc_text).tolist()
                makale_verisi = {
                    "doc_id": i,
                    "doc": d["document"],
                    "doc_text": doc_text,
                    "doc_ft_vector": doc_ft_vector,
                    "doc_bert_vector": doc_bert_vector,
                    "extractive_phrases": d["extractive_keyphrases"]
                }
                makaleler.insert_one(makale_verisi)
            except:
                pass

if ilgiler.estimated_document_count() == 0:
    print("İlgi alanları belirleniyor")
    tum_dokumanlar = []
    ilgiler_haritasi = []
    tum_ilgiler = []

    for makale in makaleler.find():
        tum_dokumanlar.extend(makale["doc"])

    kelime_sayilari = Counter(tum_dokumanlar)

    for makale in makaleler.find():
        for phrase in makale["extractive_phrases"]:
            if (phrase not in tum_ilgiler) and (phrase not in stopwords):
                tum_ilgiler.append(phrase)
                ilgiler_haritasi.append((phrase, kelime_sayilari[phrase]))

    top_ilgi_alanlari = sorted(ilgiler_haritasi, key=lambda x: x[1], reverse=True)[:20]

    for ilgi, adet in top_ilgi_alanlari:
        ilgiler.insert_one({
            "kelime": ilgi,
            "adet": adet
        })

print("Modeller yüklendi")

# Flask uygulamasını başlatma
app = Flask(__name__)


# Anasayfa
@app.route("/")
def ana_sayfa():
    ilgi_listesi = list(ilgiler.find())
    print(len(ilgi_listesi))
    return render_template("login.html", ilgi_listesi=ilgi_listesi)


# Kullanıcı kaydı
@app.route('/kayit', methods=['GET', 'POST'])
def kayit():
    if request.method == 'POST':
        degerler = {}
        degerler["kullanici_adi"] = request.form['username']
        degerler["ilgi_alanlari"] = request.form.getlist('ilgi_alanlari[]')
        degerler["sifre"] = request.form['password']

        kullanici = kullanicilar.find_one({'username': degerler["kullanici_adi"]})
        if kullanici:
            return 'Böyle bir kullanıcı zaten var.'
        else:
            kullanici_id = kullanicilar.estimated_document_count()

            kullanici_ft_vektoru = metin_to_ft_vektor(" ".join(degerler["ilgi_alanlari"])).tolist()
            kullanici_bert_vektoru = metin_to_bert_vektor(" ".join(degerler["ilgi_alanlari"])).tolist()

            kullanici_verisi = {
                'username': degerler["kullanici_adi"],
                'password': degerler["sifre"],
                'history_count': 1,
                'ilgiler': degerler["ilgi_alanlari"],
                'user_id': kullanici_id,
                'interest_ft_vector': kullanici_ft_vektoru,
                'interest_bert_vector': kullanici_bert_vektoru
            }

            kullanicilar.insert_one(kullanici_verisi)
            return f"""<html><body><h2>Kaydınız yapılmıştır.</h2><a href="/">Girişe Dön</a></body></html>"""


# Kullanıcı girişi
@app.route('/login', methods=['POST'])
def giris():
    kullanici_adi = request.form['username']
    sifre = request.form['password']

    kullanici = kullanicilar.find_one({"username": kullanici_adi})
    if not kullanici:
        return f"<html><body><h2>Kullanıcı bulunamadı.</h2><a href='/'>Girişe Dön</a></body></html>"
    elif kullanici["username"] == kullanici_adi and kullanici["password"] == sifre:
        kullanici_id = kullanici["user_id"]
        return redirect(url_for('kullanici_anasayfa', kullanici_id=kullanici_id))
    else:
        kullanici_id = kullanici["user_id"]
        return f"<html><body><h2>Başarısız giriş.</h2><a href='/'>Girişe Dön</a></body></html>"


# Kullanıcı anasayfası
@app.route('/kullanici/<int:kullanici_id>', methods=["GET", "POST"])
def kullanici_anasayfa(kullanici_id):
    kullanici_verisi = kullanicilar.find_one({"user_id": kullanici_id})

    en_benzer_ft_makaleler = benzer_fasttext_bul(kullanici_verisi["interest_ft_vector"])
    en_benzer_bert_makaleler = benzer_scibert_bul(kullanici_verisi["interest_bert_vector"])

    metrikler = precision_hesapla(kullanici_verisi, en_benzer_ft_makaleler, en_benzer_bert_makaleler)

    return render_template('kullanici.html',
                           kullanici_data=kullanici_verisi,
                           fasttext_makale=en_benzer_ft_makaleler,
                           bert_makale=en_benzer_bert_makaleler,
                           fasttext_precision=metrikler["fasttext_precision"],
                           bert_precision=metrikler["bert_precision"])


# Kullanıcı filtreleme
@app.route('/kullanici/filtre/<int:kullanici_id>', methods=["GET", "POST"])
def kullanici_filtrele(kullanici_id):
    if request.method == 'POST':
        kullanici_verisi = kullanicilar.find_one({"user_id": kullanici_id})
        filtre_sorgusu = request.form["filtre_query"]
        filtrelenmis_makaleler = []
        for makale in makaleler.find():
            if filtre_sorgusu in makale["doc_text"]:
                filtrelenmis_makaleler.append([makale])
        print(len(filtrelenmis_makaleler))

    return render_template('kullanici.html',
                           kullanici_data=kullanici_verisi,
                           fasttext_makale=filtrelenmis_makaleler[:len(filtrelenmis_makaleler) // 2],
                           bert_makale=filtrelenmis_makaleler[:len(filtrelenmis_makaleler) // 2],
                           fasttext_precision="",
                           bert_precision="")


# Kullanıcı ilgi alanı filtreleme
@app.route('/kullanici/ilgi/<int:kullanici_id>', methods=["GET", "POST"])
def kullanici_ilgi(kullanici_id):
    if request.method == 'POST':
        ilgi_sorgusu = request.form["ilgi_query"]
        kullanici_verisi = kullanicilar.find_one({"user_id": kullanici_id})
        ilgi_makaleleri = []
        for makale in makaleler.find():
            if ilgi_sorgusu in makale["extractive_phrases"]:
                ilgi_makaleleri.append([makale])

    return render_template('kullanici.html',
                           kullanici_data=kullanici_verisi,
                           fasttext_makale=ilgi_makaleleri[:len(ilgi_makaleleri) // 2],
                           bert_makale=ilgi_makaleleri[:len(ilgi_makaleleri) // 2],
                           fasttext_precision="",
                           bert_precision="")


# Makale beğenme
@app.route("/kullanici/<int:kullanici_id>/makale/<int:makale_id>")
def makale_begenildi(kullanici_id, makale_id):
    makale_verisi = makaleler.find_one({"doc_id": makale_id})
    kullanici_verisi = kullanicilar.find_one({"user_id": kullanici_id})
    kullanici_ilgileri = kullanici_verisi.get("ilgiler", [])

    for phrase in makale_verisi.get("extractive_phrases", []):
        if phrase not in kullanici_ilgileri:
            kullanici_ilgileri.append(phrase)

    for _ in range(kullanici_verisi["history_count"]):
        tum_fasttext_vektorleri = [np.array(kullanici_verisi["interest_ft_vector"])]
        tum_bert_vektorleri = [np.array(kullanici_verisi["interest_bert_vector"])]

    tum_fasttext_vektorleri.append(np.array(makale_verisi["doc_ft_vector"]))
    tum_bert_vektorleri.append(np.array(makale_verisi["doc_bert_vector"]))

    yeni_fasttext_vektoru = np.mean(tum_fasttext_vektorleri, axis=0).tolist()
    yeni_bert_vektoru = np.mean(tum_bert_vektorleri, axis=0).tolist()

    kullanicilar.update_one(
        {"user_id": kullanici_id},
        {
            "$inc": {"history_count": 1},
            "$set": {
                "interest_ft_vector": yeni_fasttext_vektoru,
                "interest_bert_vector": yeni_bert_vektoru,
                "ilgiler": kullanici_ilgileri
            }
        }
    )
    return redirect(url_for('kullanici_anasayfa', kullanici_id=kullanici_id))


# Makale görüntüleme
@app.route("/makalesayfa/<int:makale_id>")
def makale_goruntule(makale_id):
    makale = makaleler.find_one({"doc_id": makale_id})
    return render_template("makale.html", makale_data=makale)


# Benzer makaleleri bulma
def benzer_fasttext_bul(vektor):
    benzerlikler = []
    for makale in makaleler.find():
        makale_vektoru = np.array(makale["doc_ft_vector"])
        benzerlik = cosine_similarity(np.array(vektor).reshape(1, -1), makale_vektoru.reshape(1, -1))[0][0]
        benzerlikler.append((makale, benzerlik))
    sirali = sorted(benzerlikler, key=lambda x: x[1], reverse=True)
    return sirali[:5]


def benzer_scibert_bul(vektor):
    benzerlikler = []
    for makale in makaleler.find():
        makale_vektoru = np.array(makale["doc_bert_vector"])
        benzerlik = cosine_similarity(np.array(vektor).reshape(1, -1), makale_vektoru.reshape(1, -1))[0][0]
        benzerlikler.append((makale, benzerlik))
    sirali = sorted(benzerlikler, key=lambda x: x[1], reverse=True)
    return sirali[:5]


# Performans metriklerini hesaplama
def precision_hesapla(kullanici_verisi, fasttext_docs, bert_docs):
    fasttext_tp = 0
    fasttext_fp = 0
    for doc in fasttext_docs:
        if (set(kullanici_verisi["ilgiler"]) & set(doc[0]["extractive_phrases"])):
            fasttext_tp += 1
        else:
            fasttext_fp += 1

    bert_tp = 0
    bert_fp = 0
    for doc in bert_docs:
        if (set(kullanici_verisi["ilgiler"]) & set(doc[0]["extractive_phrases"])):
            bert_tp += 1
        else:
            bert_fp += 1

    return {
        "fasttext_precision": fasttext_tp / (fasttext_tp + fasttext_fp),
        "bert_precision": bert_tp / (bert_tp + bert_fp)
    }


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host="0.0.0.0", port=9000)
