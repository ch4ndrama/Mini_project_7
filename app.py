import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.cluster import KMeans

st.set_page_config(page_title="Customer Personality Predicition",page_icon="🥷",layout="centered")
st.subheader("🛒 Customer Personality prediction")
st.write("devide customers behalf of monthly spending")

st.divider()

df = pd.read_csv("Customer.csv")
with st.expander("📝 View Dataset"):
    st.dataframe(df)

x = df[["Age","Income","Online_Spending"]]

k = st.slider("Select number of clusters",min_value=2,max_value=6,value=3)

model = KMeans(n_clusters=k,random_state=42,n_init=10)
df["Clusters"] = model.fit_predict(x)

cluster_name={
    0:"🟢Budget Customer",
    1:"🟡Regular Customer",
    2:"🔴Premium Customer",
    3:"🔵VIP Customer",
    4:"🟤Luxury Customer",
    5:"🟣Elite Customer"
}
df["Customer Type"]=df["Clusters"].map(cluster_name)
st.subheader("📊 Centeres Dataset")
st.dataframe(df)

st.subheader("Customer Types")

st.dataframe(df[
    [
        "Age",
        "Income",
        "Online_Spending",
        "Clusters",
        "Customer Type"

    ]
])




st.subheader("🎃 Cluster Center")
centers = pd.DataFrame(model.cluster_centers_,columns=["Age","Income","Online_Spending"])
st.dataframe(centers)

st.success(f"Inertia Score :{model.inertia_:.2f}")

st.subheader("📈 Customer Center")

fig,ax = plt.subplots(figsize=(6,3))

scatter = ax.scatter(df["Income"],df["Online_Spending"],c=df["Clusters"],
                    cmap="viridis",s=30)

ax.scatter(model.cluster_centers_[:,1],model.cluster_centers_[:,2],
            marker="^",color="red",s=80,label="centroids")

ax.set_title("customer personality prediction")
ax.set_xlabel("Anual Income")
ax.set_ylabel("online spending")
ax.grid(True)
ax.legend()
st.pyplot(fig)