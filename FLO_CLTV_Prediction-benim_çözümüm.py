##############################################################
# BG-NBD ve Gamma-Gamma ile CLTV Prediction
##############################################################

###############################################################
# İş Problemi (Business Problem)
###############################################################

# FLO satış ve pazarlama faaliyetleri için roadmap belirlemek istemektedir.
# Şirketin orta uzun vadeli plan yapabilmesi için var olan müşterilerin gelecekte şirkete sağlayacakları potansiyel değerin tahmin edilmesi gerekmektedir.

###############################################################
# Veri Seti Hikayesi
###############################################################

# Veri seti son alışverişlerini 2020 - 2021 yıllarında OmniChannel(hem online hem offline alışveriş yapan) olarak yapan müşterilerin geçmiş alışveriş davranışlarından
# elde edilen bilgilerden oluşmaktadır.

# master_id: Eşsiz müşteri numarası
# order_channel : Alışveriş yapılan platforma ait hangi kanalın kullanıldığı (Android, ios, Desktop, Mobile, Offline)
# last_order_channel : En son alışverişin yapıldığı kanal
# first_order_date : Müşterinin yaptığı ilk alışveriş tarihi
# last_order_date : Müşterinin yaptığı son alışveriş tarihi
# last_order_date_online : Muşterinin online platformda yaptığı son alışveriş tarihi
# last_order_date_offline : Muşterinin offline platformda yaptığı son alışveriş tarihi
# order_num_total_ever_online : Müşterinin online platformda yaptığı toplam alışveriş sayısı
# order_num_total_ever_offline : Müşterinin offline'da yaptığı toplam alışveriş sayısı
# customer_value_total_ever_offline : Müşterinin offline alışverişlerinde ödediği toplam ücret
# customer_value_total_ever_online : Müşterinin online alışverişlerinde ödediği toplam ücret
# interested_in_categories_12 : Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listesi

###############################################################
# GÖREV 1: Veriyi Hazırlama
###############################################################

# 1. OmniChannel.csv verisini okuyunuz.Dataframe’in kopyasını oluşturunuz.

import pandas as pd
import datetime as dt
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)
pd.set_option("display.float_format", lambda x: '%.3f' % x)

df_ = pd.read_csv("flo_data_20k.csv")
omnidf = df_.copy()
omnidf.head()
omnidf.describe().T
omnidf.isnull().sum()

# 2. Aykırı değerleri baskılamak için gerekli olan outlier_thresholds ve replace_with_thresholds fonksiyonlarını tanımlayınız.
# Not: cltv hesaplanırken frequency değerleri integer olması gerekmektedir.Bu nedenle alt ve üst limitlerini round() ile yuvarlayınız.

def outlier_thresholds(dataframe, variable, lower_quantile=0.01, upper_quantile=0.99):
    """
    Belirtilen değişken için alt ve üst aykırı değer sınırlarını döndürür.
    Varsayılan olarak 1. ve 99. yüzdelikler kullanılır.
    """
    q1 = dataframe[variable].quantile(lower_quantile)
    q3 = dataframe[variable].quantile(upper_quantile)
    iqr = q3 - q1
    lower_limit = q1 - 1.5 * iqr
    upper_limit = q3 + 1.5 * iqr
    return lower_limit, upper_limit

def replace_with_thresholds(dataframe, variable, lower_quantile=0.01, upper_quantile=0.99, round_freq=False):
    """
    Aykırı değerleri belirlenen alt ve üst sınırlar ile baskılar.
    frequency değişkeni için round() uygulanır.
    """
    lower_limit, upper_limit = outlier_thresholds(dataframe, variable, lower_quantile, upper_quantile)

    # Baskılama işlemi
    dataframe.loc[dataframe[variable] < lower_limit, variable] = lower_limit
    dataframe.loc[dataframe[variable] > upper_limit, variable] = upper_limit

    # frequency değişkeni integer olmalı
    if round_freq:
        dataframe[variable] = dataframe[variable].round().astype(int)

# 3. "order_num_total_ever_online","order_num_total_ever_offline","customer_value_total_ever_offline","customer_value_total_ever_online" değişkenlerinin
#aykırı değerleri varsa baskılayanız.

columns = ["order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline","customer_value_total_ever_online"]
for col in columns:
    replace_with_thresholds(omnidf, col)

# 4. Omnichannel müşterilerin hem online'dan hemde offline platformlardan alışveriş yaptığını ifade etmektedir.
# Herbir müşterinin toplam alışveriş sayısı ve harcaması için yeni değişkenler oluşturun.

omnidf["total_order_number"] = omnidf["order_num_total_ever_online"] + omnidf["order_num_total_ever_offline"]
omnidf.groupby(["master_id"]).agg({"total_order_number": "sum"})
omnidf["total_customer_value"] = omnidf["customer_value_total_ever_offline"] + omnidf["customer_value_total_ever_online"]
omnidf.head()

# 5. Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.

print(omnidf.dtypes)

#1. yol

for col in omnidf.columns:
    if "date" in col:
        omnidf[col] = pd.to_datetime(omnidf[col])

#2. yol

date_columns = omnidf.columns[omnidf.columns.str.contains("date")]
omnidf[date_columns] = omnidf[date_columns].apply(pd.to_datetime)

###############################################################
# GÖREV 2: CLTV Veri Yapısının Oluşturulması
###############################################################

# 1.Veri setindeki en son alışverişin yapıldığı tarihten 2 gün sonrasını analiz tarihi olarak alınız.

omnidf.sort_values("last_order_date", ascending = False)
today_date = dt.datetime(2021, 6, 1 )
type(today_date)

# 2.customer_id, recency_cltv_weekly, T_weekly, frequency ve monetary_cltv_avg değerlerinin yer aldığı yeni bir cltv dataframe'i oluşturunuz.

#customer_id
cltv_df = pd.DataFrame()
cltv_df["customer_id"] = omnidf["master_id"]

#recency_cltv_weekly, T_weekly
#---> son satin alma uzerinden gecen zamani haftalik olarak verir, haftalik olarak
cltv_df["recency_cltv_weekly"] = ((omnidf["last_order_date"] - omnidf["first_order_date"]).dt.days) / 7
#---->Tenor, musterinin yasi. Analiz tarihinden ne kadar sure once ilk satin alma yapilmis
cltv_df["T_weekly"] = ((today_date - omnidf["first_order_date"]).dt.days) / 7

#frequency ----> Tekrar eden toplam satin alma sayisi
cltv_df["frequency"] = omnidf["total_order_number"]

#monetary_cltv_avg -----> Satin alma basina ortalama kazanc
cltv_df["monetary_cltv_avg"] = omnidf["total_customer_value"] / omnidf["total_order_number"]

cltv_df.head()

###############################################################
# GÖREV 3: BG/NBD, Gamma-Gamma Modellerinin Kurulması, 6 aylık CLTV'nin hesaplanması
###############################################################

# 1. BG/NBD modelini kurunuz.

from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter

cdf = cltv_df.copy()

# Enforce numeric
for c in ["frequency","recency_cltv_weekly","T_weekly","monetary_cltv_avg"]:
    cdf[c] = pd.to_numeric(cdf[c], errors="coerce")

# Required by BG/NBD & GG
cdf["frequency"] = cdf["frequency"].round().astype("Int64")  # integer for frequency
cdf = cdf[
    (cdf["frequency"] > 0) &                              # fit only on purchasers
    (cdf["recency_cltv_weekly"] >= 0) &
    (cdf["T_weekly"] >= cdf["recency_cltv_weekly"]) &
    (~cdf[["frequency","recency_cltv_weekly","T_weekly"]].isna().any(axis=1)) &
    np.isfinite(cdf[["frequency","recency_cltv_weekly","T_weekly"]]).all(axis=1)
].copy()

# Drop edge cases that often break the likelihood
cdf = cdf[(cdf["T_weekly"] > 0)]                # avoid zero calibration window
cdf = cdf[~((cdf["recency_cltv_weekly"] == 0) & (cdf["frequency"] == 1))]  # pathological singletons

# Cap extreme frequency tails (protect optimizer)
q99 = cdf["frequency"].quantile(0.99)
cdf.loc[cdf["frequency"] > q99, "frequency"] = q99
cdf["frequency"] = cdf["frequency"].round().astype(int)

old = np.seterr(invalid="raise", divide="raise", over="raise")

bgf = BetaGeoFitter(penalizer_coef=0.01)  # biraz daha kuvvetli düzenleme
bgf.fit(
    cdf["frequency"],
    cdf["recency_cltv_weekly"].astype(float),
    cdf["T_weekly"].astype(float)
)

print(bgf)  # a,b > 0 olmalı
np.seterr(**old)

print(cdf[["frequency","recency_cltv_weekly","T_weekly"]].describe(percentiles=[.5,.9,.99]))

# 3 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_3_month olarak cltv dataframe'ine ekleyiniz.

cdf["exp_sales_3_month"] = bgf.predict(4*3,
                                       cdf['frequency'],
                                       cdf['recency_cltv_weekly'],
                                       cdf['T_weekly'])

# 6 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_6_month olarak cltv dataframe'ine ekleyiniz.

cdf["exp_sales_6_month"] = bgf.predict(4*6,
                                       cdf['frequency'],
                                       cdf['recency_cltv_weekly'],
                                       cdf['T_weekly'])

# 3. ve 6.aydaki en çok satın alım gerçekleştirecek 10 kişiyi inceleyeniz.

#1. yol
cdf.sort_values(by=["exp_sales_3_month","exp_sales_6_month"], ascending=False).head(10)

#2. yol
cdf.sort_values("exp_sales_3_month",ascending=False)[:10]
cdf.sort_values("exp_sales_6_month",ascending=False)[:10]

# 2.  Gamma-Gamma modelini fit ediniz. Müşterilerin ortalama bırakacakları değeri tahminleyip exp_average_value olarak cltv dataframe'ine ekleyiniz.

ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cdf['frequency'], cdf['monetary_cltv_avg'])
cdf["exp_average_value"] = ggf.conditional_expected_average_profit(cdf['frequency'],
                                                                cdf['monetary_cltv_avg'])
cdf.head()

# 3. 6 aylık CLTV hesaplayınız ve cltv ismiyle dataframe'e ekleyiniz.

cltv = ggf.customer_lifetime_value(bgf,
                                   cdf['frequency'],
                                   cdf['recency_cltv_weekly'],
                                   cdf['T_weekly'],
                                   cdf['monetary_cltv_avg'],       #Gamma-Gamma modelinin girdisidir.
                                   time=24,                             #Zaman periyodu, next 24 hafta icin
                                   freq="W",                           #Zaman birimi =week
                                   discount_rate=0.01)                 #Gelecekteki gelirleri bugüne indirgenmiş (discounted) değer olarak gösterir.
cdf["cltv"] = cltv

# CLTV değeri en yüksek 20 kişiyi gözlemleyiniz.

cdf.sort_values(by = "cltv", ascending = False).head(20)

###############################################################
# GÖREV 4: CLTV'ye Göre Segmentlerin Oluşturulması
###############################################################

# 1. 6 aylık CLTV'ye göre tüm müşterilerinizi 4 gruba (segmente) ayırınız ve grup isimlerini veri setine ekleyiniz.
# cltv_segment ismi ile atayınız.

cdf["cltv_segment"] = pd.qcut(cdf["cltv"], 4, labels = ["D", "C", "B", "A"])
cdf.head

# 2. Segmentlerin recency, frequnecy ve monetary ortalamalarını inceleyiniz.

cdf.groupby("cltv_segment").agg({"recency_cltv_weekly": "mean",
                                 "frequency": "mean",
                                 "monetary_cltv_avg": "mean"
                                 })






