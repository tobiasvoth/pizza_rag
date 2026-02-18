import chromadb
import pandas as pd
import umap
import plotly.express as px

# 1. Verbindung herstellen
client = chromadb.PersistentClient(path="./chrome_langchain_db") 
collection = client.get_collection(name="restaurant_reviews")

# 2. Daten abrufen (Wichtig: 'metadatas' inkludieren, 'ids' weglassen)
results = collection.get(include=['embeddings', 'documents', 'metadatas'])

if results['embeddings'] is None or len(results['embeddings']) == 0:
    print("Keine Daten gefunden.")
else:
    # 3. UMAP auf 3 Dimensionen
    print("Berechne 3D-Projektion...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=3, random_state=42)
    projection = reducer.fit_transform(results['embeddings'])

    # 4. Ratings aus den Metadaten extrahieren
    # Wir nutzen eine List-Comprehension, um das Feld 'rating' sicher auszulesen
    # Falls ein Dokument kein Rating hat, setzen wir 0 als Standardwert
    ratings = [m.get('rating', 0) if m else 0 for m in results['metadatas']]

    # 5. DataFrame erstellen
    df = pd.DataFrame({
        'x': projection[:, 0],
        'y': projection[:, 1],
        'z': projection[:, 2],
        'text': results['documents'],
        'rating': ratings,
        'id': results['ids']
    })

    # 6. 3D-Plot mit Farbskala
    fig = px.scatter_3d(
        df, x='x', y='y', z='z',
        color='rating',                 # Hier wird das Rating für die Farbe genutzt
        color_continuous_scale='RdYlGn', # 'RdYlGn' (Rot-Gelb-Grün) wäre auch passend
        hover_data={'text': True, 'rating': True, 'x': False, 'y': False, 'z': False},
        title='ChromaDB 3D Explorer (Farbe = Rating 0-5)',
        labels={'rating': 'Bewertung', 'text': 'Inhalt'},
        opacity=0.8
    )

    # Styling
    fig.update_traces(marker=dict(size=4))
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=40))

    fig.show()