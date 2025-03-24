import streamlit as st
import json
import pandas as pd
import matplotlib.pyplot as plt
import ollama


def load_experiment_data():
    """
    Load experiment data from data.json.
    
    Returns:
        pandas.DataFrame: Experiments data
    """
    try:
        with open('data.json', 'r') as f:
            data = json.load(f)
        
        # Convert to DataFrame for easy manipulation
        df = pd.DataFrame([
            {**exp['hyperparameters'], **exp['metrics'], 'experiment_name': exp['experiment_name']}
            for exp in data
        ])
        
        return df, data
    except Exception as e:
        st.error(f"Error loading experiment data: {e}")
        return pd.DataFrame(), []

def main():
    st.title('TorchTrack Experiment Tracker')
    
    # Load data
    df, full_data = load_experiment_data()
    
    if df.empty:
        st.write("No experiment data available.")
        return
    
    # Sort DataFrame by performance metric (assuming 'accuracy' or 'r2_score')
    metric_cols = [col for col in df.columns if col in ['accuracy', 'r2_score']]
    if metric_cols:
        metric_col = metric_cols[0]
        df_sorted = df.sort_values(by=metric_col, ascending=False)
    else:
        df_sorted = df
    
    # Display sorted table
    st.subheader('Experiment Results')
    st.dataframe(df_sorted)
    
    # Epoch-wise Plots
    st.subheader('Epoch-wise Performance')
    
    # Plotting for each experiment
    for exp in full_data:
        st.write(f"Experiment: {exp['experiment_name']}")
        
        # Create columns for loss and accuracy plots
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Loss per Epoch")
            loss_data = exp['epoch_data']['loss_per_epoch']
            st.line_chart(loss_data)
        
        with col2:
            st.write("Accuracy per Epoch")
            accuracy_data = exp['epoch_data']['accuracy_per_epoch']
            st.line_chart(accuracy_data)

    with st.sidebar:
        st.header("AI analysis")
        with st.expander("Model performance analysis"):
            response = ollama.chat(
                model="llama3.2",
                messages=[{
                    "role": "user", 
                    "content": f"""
                    You are an expert on analyzing ml models, 
                    Generate an in detail performance analysis based on the info I give you:
                    The model is a {exp["model_type"]} model with given params and results: 
                    {df_sorted.to_string()},
                    Model data:
                    {exp["model_data"]}
                    """
                }], stream=True
            )
            res_content = ""
            def catch_response(response):
                global response_content
                for chunck in response:
                    response_content = chunck["message"]["content"]
                    yield chunck["message"]["content"]
            
            stream = catch_response(response)
            st.write_stream(stream)
        
        with st.expander("Architechture recommendation"):
            response = ollama.chat(
                model="llama3.2",
                messages=[{
                    "role": "user", 
                    "content": f"""
                    You are an expert on analyzing ml models, 
                    Generate an in detail architechture recommendation based on the info I give you:
                    The model is a {exp["model_type"]} model with given params and results: 
                    {df_sorted.to_string()},
                    Model data:
                    {exp["model_data"]}
                    """
                }], stream=True
            )
            res_content = ""
            def catch_response(response):
                global response_content
                for chunck in response:
                    response_content = chunck["message"]["content"]
                    yield chunck["message"]["content"]
            
            stream = catch_response(response)
            st.write_stream(stream)

        with st.expander("Training optimization"):
            response = ollama.chat(
                model="llama3.2",
                messages=[{
                    "role": "user", 
                    "content": f"""
                    You are an expert on analyzing ml models, 
                    Generate an in detail training optimization recommendations based on the info I give you:
                    The model is a {exp["model_type"]} model with given params and results: 
                    {df_sorted.to_string()},
                    Model data:
                    {exp["model_data"]}
                    """
                }], stream=True
            )
            res_content = ""
            def catch_response(response):
                global response_content
                for chunck in response:
                    response_content = chunck["message"]["content"]
                    yield chunck["message"]["content"]
            
            stream = catch_response(response)
            st.write_stream(stream)

if __name__ == '__main__':
    main()