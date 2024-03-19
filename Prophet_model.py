import pandas as pd
import matplotlib.pyplot as plt
from plotly.offline import iplot
import seaborn as sns
from plotly import tools
import plotly.graph_objs as go
import gc

from datetime import datetime

# Prophet model
from prophet import Prophet

# Fonction de preprocessing
def preprocessing(df2, code_dep, delit):
    """
    Input:
    df2 : Dataframe d'entrée contenant les données brutes
    code_dep : Code du département spécifié
    delit : Type de délit spécifié

    Output:
    df : Dataframe filtré contenant uniquement les colonnes 'Date' et le type de délit spécifié
    """

    # Convertir la colonne de dates au format datetime
    df2['Date'] = pd.to_datetime(df2['Date'], dayfirst=True)  # Convertit la colonne 'Date' au format datetime

    # Filtrer le dataset pour le code département spécifié
    df = df2[df2[
                 'code_dep'] == code_dep]  # Filtre le dataframe pour ne garder que les lignes avec le code de département spécifié

    # Supprimer les colonnes autres que la date et le type de délit spécifié
    df = df.drop(
        columns=df.columns.difference(['Date', delit]))  # Supprime toutes les colonnes sauf la date et le type de délit

    return df  # Renvoie le dataframe filtré

class prophet_model:
    def __init__(self,df,code_dep, delit, periode):
        self.df = df
        self.code_dep = code_dep
        self.delit = delit
        self.periode = periode

    def serie_temporelle(self):
        """
        Input:
        df : Dataframe filtré
        code_dep : Code du département spécifié
        delit : Type de délit spécifié

        Output:
        None : Visualisation de la série temporelle

        """

        g1 = go.Scatter(
            x=self.df['Date'],
            y=self.df[self.delit],
            mode='lines',
            name=self.delit)

        layout = dict(
            title=f"{self.delit} dans le département {self.code_dep} entre 1999 et 2200",
            xaxis=dict(
                rangeslider=dict(visible=True),
                type='date')
        )
        data = [g1]
        fig = dict(data=data, layout=layout)
        iplot(fig, filename="Time Series with Rangeslider")

    def split_data(self):
        """
        Input :
        df : Dataframe filtré
        periode : Période sur laquelle on base la séparation des données, elle correspond à la période sur laquelle on va prédire également

        Output :
        df_train : Données d'entrainement
        df_test : Données de test

        """
        # Création des données de test et d'entrainement
        df_train = self.df.iloc[self.periode:, :]
        df_test = self.df.iloc[0:self.periode, :]
        return df_train, df_test

    def entrainement(self,df_train, df_test):
        """
        Input:
        df_train : Données d'entrainement
        df_test : Données de test
        periode : Période à choisir sur laquelle on va prédire

        Output:
        prophet_model : Le modèle entrainé
        forecast : Les données de prévision

        """

        # Ajustement du nom des colonnes pour faire fonctionner Prophet
        df_model = df_train.reset_index().rename(columns={'Date': 'ds', self.delit: 'y'})

        # Création et entrainement du modèle prophet
        prophet_model = Prophet()
        prophet_model.fit(df_model)

        # Périodisation et définition de la plage temporelle
        future = prophet_model.make_future_dataframe(periods=self.periode, freq='M')

        # Prédictions et affichage
        forecast = prophet_model.predict(df=future)
        return (prophet_model, forecast)

    def prevision(self,prophet_model,forecast,df_test):
        """
        Input:
        prophet_model : Modèle entrainé
        forecast : Prévisions
        code_dep : Code du département spécifié
        delit : Type de délit spécifié
        periode : Période à choisir sur laquelle on va prédire

        Output:
        None : Affichage des observations et des prévisions avec comparaison aux données réelles

        """
        # Tracer le graphique avec les légendes
        fig1 = prophet_model.plot(forecast)
        plt.plot(df_test['Date'], df_test[self.delit], color='red')
        plt.xlabel('Date')  # Ajouter une légende pour l'axe x
        plt.ylabel(self.delit)  # Ajouter une légende pour l'axe y
        plt.title(
            f'Prévisions des {self.delit} à horizon de {self.periode} mois dans le département {self.code_dep}')  # Ajouter un titre au graphique
        plt.legend(['Observations', 'Prévisions', 'Incertitude',
                    'Données réelles'])  # Ajouter une légende pour chaque série de données

        # Changer la couleur du quadrillage en blanc
        plt.grid(color='black')
        plt.show()  # Afficher le graphique
        plt.close()

    def tendance(self,prophet_model,forecast):
        """
        Input:
        prophet_model : Modèle entrainé
        forecast : Prévisions

        Output:
        None : Affichage de la tendance et de la saisonnalité du modèle

        """
        # Tracer les composantes saisonnières
        fig_seasonal = prophet_model.plot_components(forecast)
        plt.title("Tendance et saisonnalité du modèle")
        # Afficher le graphique avec les légendes
        plt.show()
        plt.close()






