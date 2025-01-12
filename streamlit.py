import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
import datetime
import asyncio
import httpx
from joblib import Parallel, delayed

def load(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values(by=["city", "timestamp"])
        return df
    else:
        return None

def seasons(df):
    return (
        df.groupby(["city", "season"])["temperature"]
          .agg(mean_temp="mean", std_temp="std")
          .reset_index()
    )

def anomalies(df, season_stats):
    merged = pd.merge(df, season_stats, how="left", on=["city", "season"])
    lower_bound = merged["mean_temp"] - 2 * merged["std_temp"]
    upper_bound = merged["mean_temp"] + 2 * merged["std_temp"]
    merged["is_anomaly"] = ~merged["temperature"].between(lower_bound, upper_bound)
    return merged

def rolling(df, window=30):
    df = df.copy()
    df["rolling_mean"] = (
        df.groupby("city")["temperature"]
          .transform(lambda x: x.rolling(window, min_periods=1).mean())
    )
    df["rolling_std"] = (
        df.groupby("city")["temperature"]
          .transform(lambda x: x.rolling(window, min_periods=1).std())
    )
    return df

def season(dt):
    m = dt.month
    d = {12:"winter",1:"winter",2:"winter",3:"spring",4:"spring",5:"spring",
        6:"summer",7:"summer",8:"summer",9:"autumn",10:"autumn",11:"autumn"}
    return d[m]

def weather_sync(city, api_key):
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"q": city, "appid": api_key, "units": "metric"}
    r = requests.get(url, params=params)
    return r.json()

async def weather_async(city, api_key):
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"q": city, "appid": api_key, "units": "metric"}
    async with httpx.AsyncClient() as c:
        r = await c.get(url, params=params)
        return r.json()

def normal(city, current_temp, stats):
    s = season(datetime.datetime.now())
    row = stats.query("city == @city and season == @s")
    if row.empty:
        return True
    m = row["mean_temp"].values[0]
    sd = row["std_temp"].values[0]
    lb = m - 2*sd
    ub = m + 2*sd
    return lb <= current_temp <= ub

def main():
    st.title("Анализ исторических температур и мониторинг текущей погоды")
    st.sidebar.header("Шаг 1: Загрузите CSV-файл")
    uploaded_file = st.sidebar.file_uploader("CSV-файл (columns: city, timestamp, temperature, season)", type=["csv"])
    
    df = load(uploaded_file)
    if df is None or df.empty:
        st.warning("Необходимо загрузить корректный CSV-файл.")
        st.stop()
    
    cities = sorted(df["city"].unique().tolist())
    st.sidebar.header("Шаг 2: Выберите город")
    chosen_city = st.sidebar.selectbox("Город", cities)

    st.sidebar.header("Шаг 3: Введите OpenWeatherMap API Key")
    key = st.sidebar.text_input("API Key", value="", type="password")

    st.subheader(f"Исторические данные для города {chosen_city}")

    city_data = df[df["city"] == chosen_city].copy()
    season_stats = seasons(df)
    df_an = anomalies(df, season_stats)
    city_an = df_an[df_an["city"] == chosen_city]
    city_roll = rolling(city_data)

    st.write("**Описательная статистика:**")
    st.write(city_data["temperature"].describe())

    fig_time = px.scatter(city_an, x="timestamp", y="temperature", color="is_anomaly",
                          title=f"Температура в {chosen_city} (аномалии)")
    st.plotly_chart(fig_time, use_container_width=True)

    fig_roll = px.line(city_roll, x="timestamp", y="rolling_mean",
                       title=f"Скользящее среднее (30 дн.) в {chosen_city}")
    st.plotly_chart(fig_roll, use_container_width=True)

    cs = season_stats[season_stats["city"] == chosen_city]
    fig_season = px.bar(cs, x="season", y="mean_temp", error_y="std_temp",
                        title=f"Сезонные статистики в {chosen_city}")
    st.plotly_chart(fig_season, use_container_width=True)

    st.subheader("Текущая погода")
    if key:
        sw = weather_sync(chosen_city, key)
        if "cod" in sw and sw["cod"] == 401:
            st.error(sw)
        elif "main" not in sw:
            st.warning("Не удалось получить данные о погоде.")
        else:
            cur_t = sw["main"]["temp"]
            st.write(f"Синхронный запрос: {chosen_city} = {cur_t:.2f} °C")
            if normal(chosen_city, cur_t, season_stats):
                st.success("Температура в норме.")
            else:
                st.error("Температура аномальна.")
            aw = asyncio.run(weather_async(chosen_city, key))
            if "cod" in aw and aw["cod"] == 401:
                st.error(aw)
            elif "main" not in aw:
                st.warning("Не удалось получить данные (async).")
            else:
                at = aw["main"]["temp"]
                st.write(f"Асинхронный запрос: {chosen_city} = {at:.2f} °C")
                if normal(chosen_city, at, season_stats):
                    st.success("Температура в норме (async).")
                else:
                    st.error("Температура аномальна (async).")
    else:
        st.info("Введите API Key для просмотра текущей погоды.")

if __name__ == "__main__":
    main()