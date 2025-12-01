import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
import os
import re
from transformers import pipeline, AutoTokenizer

# ==============================
#   FONCTIONS UTILITAIRES
# ==============================

def nettoyer_texte(texte):
    """Nettoyage des textes"""
    if pd.isna(texte):
        return ""
    texte = str(texte).lower()
    texte = re.sub(r'http\S+|www\S+', '', texte)
    texte = re.sub(r'[^\w\s\']', ' ', texte)
    texte = re.sub(r'\s+', ' ', texte)
    return texte.strip()

@st.cache_resource
def charger_modele_sentiment():
    """Charge le modèle Transformer depuis HuggingFace avec cache local"""
    try:
        # Le modèle sera téléchargé une fois et mis en cache automatiquement
        sentiment_model = pipeline(
            "sentiment-analysis",
            model="jgmagarino/tourist-comments-classifier",
            truncation=True,
            tokenizer=AutoTokenizer.from_pretrained(
                "jgmagarino/tourist-comments-classifier",
                truncation=True,
                padding=True,
                max_length=512
            ),
        )
        st.success("Modèle chargé avec succès")
        return sentiment_model
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle: {e}")
        return None

# Charger le modèle sentiment
sentiment_model = charger_modele_sentiment()

# ==============================
#   TITRE
# ==============================
st.title(" Analyse de Sentiments")


# ==============================
#   CHARGEMENT DATASET
# ==============================
df = pd.read_csv('analyse_sentiments_complete.csv', sep=';')

# Conversion label -> texte
if 'label' in df.columns:
    df['transformer_label'] = df['label'].map({1: "POSITIF", 0: "NEGATIF"})

# ==============================
#   ONGLET : PREDICTION
# ==============================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    " Prédiction",
    " Aperçu du Dataset",
    " Distribution des Sentiments",
    " Nuage de Mots",
    " Analyse par Mots-Clés"
])

# ------ ONGLET 1 : PREDICTION ------
with tab1:
    texte_input = st.text_area("📝 Saisissez un avis à analyser :", "")

    if st.button("Prédire le sentiment"):
        if not texte_input.strip():
            st.warning("Veuillez saisir un texte à analyser !")
        else:
            texte_clean = nettoyer_texte(texte_input)
            resultat = sentiment_model([texte_clean])[0]
            sentiment = "POSITIF" if resultat["label"] == "LABEL_1" else "NÉGATIF"

            # Couleurs selon sentiment
            if sentiment == "POSITIF":
                st.success("🌟 **POSITIF**")
            else:
                st.error("🔴 **NÉGATIF**")

            st.write(f"Confiance : **{resultat['score']*100:.1f}%**")

            # Progress bar colorée selon score
            st.progress(resultat["score"])


# ------ ONGLET 2 : APERÇU DATASET ------
with tab2:
    st.write("###  Aperçu des premières lignes du dataset")
    st.dataframe(df[['titre_avis', 'texte_avis', 'note_avis', 'date_avis', 'type_voyage', 'nb_mots', 'nb_phrases']].head())

    st.write("###  Informations")
    st.write(df[['titre_avis', 'texte_avis', 'note_avis', 'date_avis', 'type_voyage', 'nb_mots', 'nb_phrases']].describe(include='all'))


# ------ ONGLET 3 : DISTRIBUTION ------
with tab3:
    st.write("###  Distribution des Sentiments")

    fig, ax = plt.subplots()

    couleurs = ['#2ecc71', '#e74c3c']  # vert / rouge
    # Mapper LABEL_0 et LABEL_1 vers NEGATIF et POSITIF
    label_mapping = {'LABEL_0': "NEGATIF", 'LABEL_1': "POSITIF"}
    df['label_text'] = df['transformer_label'].map(label_mapping)
    sentiment_counts = df['label_text'].value_counts()
    sentiment_counts.plot(kind='bar', ax=ax, color=couleurs)
    
    ax.set_xticklabels(sentiment_counts.index, rotation=0)

    plt.xlabel("Sentiment")
    plt.ylabel("Nombre d'avis")
    plt.title("Répartition des sentiments")

    st.pyplot(fig)

    # Top 5 avis positifs et négatifs
    st.write("---")
    st.write("### Top 5 Avis par Sentiment")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("#### Top 5 Avis Positifs")
        df_positifs = df[df['label_text'] == 'POSITIF'].nlargest(5, 'transformer_score')[
            ['titre_avis', 'texte_avis', 'note_avis', 'date_avis', 'type_voyage', 'label_text', 'transformer_score']
        ]
        df_positifs = df_positifs.reset_index(drop=True)
        df_positifs.index = df_positifs.index + 1
        df_positifs_display = df_positifs.copy()
        df_positifs_display.columns = ['Titre', 'Texte', 'Note', 'Date', 'Type Voyage', 'Sentiment', 'Score']
        st.dataframe(df_positifs_display, use_container_width=True)
    
    with col2:
        st.write("#### Top 5 Avis Négatifs")
        df_negatifs = df[df['label_text'] == 'NEGATIF'].nlargest(5, 'transformer_score')[
            ['titre_avis', 'texte_avis', 'note_avis', 'date_avis', 'type_voyage', 'label_text', 'transformer_score']
        ]
        df_negatifs = df_negatifs.reset_index(drop=True)
        df_negatifs.index = df_negatifs.index + 1
        df_negatifs_display = df_negatifs.copy()
        df_negatifs_display.columns = ['Titre', 'Texte', 'Note', 'Date', 'Type Voyage', 'Sentiment', 'Score']
        st.dataframe(df_negatifs_display, use_container_width=True)


# ------ ONGLET 4 : WORDCLOUD ------
with tab4:
    st.write("###  Nuage de Mots")

    texte_total = " ".join(df['texte_avis'].astype(str))

    wc = WordCloud(
        width=800,
        height=400,
        background_color='white',
        colormap='viridis'   # colormap stylée
    ).generate(texte_total)

    fig, ax = plt.subplots(figsize=(10,5))
    ax.imshow(wc)
    ax.axis("off")
    st.pyplot(fig)


# ------ ONGLET 5 : ANALYSE PAR MOTS-CLÉS ------
with tab5:
    st.write("### Analyse par mot-clé")

    mot = st.text_input("Entrez un mot à rechercher :")

    if mot:
        resultats = df[df['texte_avis'].str.contains(mot, case=False, na=False)]

        st.write(f"### Résultats pour **{mot}** : {len(resultats)} avis")

        # Sélectionner et renommer les colonnes demandées
        colonnes_selectionnees = [
            'titre_avis', 'texte_avis', 'note_avis', 'date_avis', 'type_voyage',
            'nb_mots', 'nb_phrases', 'label_text', 'transformer_score'
        ]
        resultats_display = resultats[colonnes_selectionnees].copy()
        resultats_display.columns = ['Titre', 'Texte', 'Note', 'Date', 'Type Voyage', 
                                      'Nb Mots', 'Nb Phrases', 'Sentiment', 'Score']
        
        # Style couleur sur la colonne sentiment
        resultats_style = resultats_display.style.apply(
            lambda s: ['background-color: #2ecc7055' if v=='POSITIF' else 'background-color: #e74c3c55' for v in s],
            subset=['Sentiment']
        )

        st.dataframe(resultats_style, use_container_width=True)
