import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd


def show_prediction(prob_log,prob_knn,prob_rf,prob_svm,log_model,input_values, log_metrics, knn_metrics, rf_metrics, svm_metrics):
    st.subheader("Prdeicion Result")
    colA, colB, colC, colD = st.columns(4)
    colA.metric("Logistic Probability :",f"{prob_log*100:.2f}")
    colB.metric("KNN Probability :",f"{prob_knn*100:.2f}")
    colC.metric("Random Forest Probability :",f"{prob_rf*100:.2f}")
    colD.metric("SVM Probability :",f"{prob_svm*100:.2f}")
    
    st.subheader("Model Accuracy (Train vs Test)")
    df_acc = pd.DataFrame({
        "Model": ["Logistic","KNN","Random Forest","SVM"],
        "Train Accuracy" : [log_metrics["train_accuracy"],knn_metrics["train_accuracy"],rf_metrics["train_accuracy"],svm_metrics["train_accuracy"]],
        "Test Accuracy" : [log_metrics["test_accuracy"],knn_metrics["test_accuracy"],rf_metrics["test_accuracy"],svm_metrics["test_accuracy"]]
    })
    st.dataframe(df_acc)
    
    if prob_log > 0.5:
        st.success("StartUp likely to succed")
    else :
        st.error("High risk of failure")

    col1,col2=st.columns(2)
    ## --------------LEFT : PREDICTION 
    with col1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob_log * 100,
            title={'text' : "Success Probability"},
            gauge={'axis': {'range': [0, 100]}}
        ))
        st.plotly_chart(fig)

    ## --------------RIGHT : PREDICTION
    with col2:
        st.subheader("Feature Importance")
        features = ["Experience","Team","Funding","Market","Innovation","Marketing","Competition","Industry","Education","Stage","Revenue"]
        importance = log_model.coef_[0]
        df_img = pd.DataFrame({
            "Feature": features,
            "Importance": importance
        })
        fig2 = px.bar(df_img, x="Importance", y="Feature", orientation="h")
        st.plotly_chart(fig2)