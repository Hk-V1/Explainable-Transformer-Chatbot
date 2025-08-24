import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    TrainingArguments, Trainer, 
    TextDataset, DataCollatorForLanguageModeling
)
from datasets import Dataset
import json
import os
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

# Set up page config
st.set_page_config(
    page_title="Explainable Transformer Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

class ExplainableTransformerChatbot:
    """
    A transformer-based chatbot with explainability features including
    attention visualization and token probability analysis.
    """
    
    def __init__(self, model_name: str = "distilgpt2"):
        """
        Initialize the chatbot with a pretrained model.
        
        Args:
            model_name: Name of the pretrained model to use
        """
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            output_attentions=True,
            output_hidden_states=True
        )
        
        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model.to(self.device)
        self.model.eval()
        
    def fine_tune_model(self, dataset_text: str, output_dir: str = "./fine_tuned_model",
                       num_epochs: int = 3, learning_rate: float = 5e-5) -> bool:
        """
        Fine-tune the model on custom dataset.
        
        Args:
            dataset_text: Text data for fine-tuning
            output_dir: Directory to save the fine-tuned model
            num_epochs: Number of training epochs
            learning_rate: Learning rate for training
            
        Returns:
            bool: Success status
        """
        try:
            # Prepare dataset
            lines = [line.strip() for line in dataset_text.split('\n') if line.strip()]
            dataset = Dataset.from_dict({"text": lines})
            
            def tokenize_function(examples):
                return self.tokenizer(
                    examples["text"], 
                    truncation=True, 
                    padding=True, 
                    max_length=512
                )
            
            tokenized_dataset = dataset.map(tokenize_function, batched=True)
            
            # Data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            )
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir=output_dir,
                overwrite_output_dir=True,
                num_train_epochs=num_epochs,
                per_device_train_batch_size=4,
                save_steps=10_000,
                save_total_limit=2,
                prediction_loss_only=True,
                learning_rate=learning_rate,
                warmup_steps=100,
                logging_steps=100,
            )
            
            # Trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                data_collator=data_collator,
                train_dataset=tokenized_dataset,
            )
            
            # Train
            trainer.train()
            trainer.save_model()
            self.tokenizer.save_pretrained(output_dir)
            
            return True
            
        except Exception as e:
            st.error(f"Fine-tuning failed: {str(e)}")
            return False
    
    def load_fine_tuned_model(self, model_path: str) -> bool:
        """
        Load a fine-tuned model.
        
        Args:
            model_path: Path to the fine-tuned model
            
        Returns:
            bool: Success status
        """
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                output_attentions=True,
                output_hidden_states=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.to(self.device)
            self.model.eval()
            return True
        except Exception as e:
            st.error(f"Failed to load model: {str(e)}")
            return False
    
    def generate_response(self, prompt: str, temperature: float = 0.7,
                         top_k: int = 50, top_p: float = 0.95,
                         max_length: int = 100) -> Dict:
        """
        Generate response with detailed information for explainability.
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            max_length: Maximum generation length
            
        Returns:
            Dict containing response and analysis data
        """
        # Tokenize input
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        input_length = input_ids.shape[1]
        
        # Generate with detailed outputs
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=True,
                return_dict_in_generate=True,
                output_attentions=True,
                output_scores=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Extract generated sequence
        generated_ids = outputs.sequences[0]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        response_text = generated_text[len(prompt):].strip()
        
        # Get tokens and their IDs
        all_tokens = self.tokenizer.convert_ids_to_tokens(generated_ids)
        all_token_ids = generated_ids.cpu().numpy().tolist()
        
        # Get attention weights (from last layer for simplicity)
        attentions = outputs.attentions[-1]  # Last generation step
        attention_weights = attentions[0].cpu().numpy()  # First batch item
        
        # Get token probabilities for generated tokens
        token_probs = []
        if outputs.scores:
            for score in outputs.scores:
                probs = F.softmax(score[0], dim=-1).cpu().numpy()
                token_probs.append(probs)
        
        return {
            "response": response_text,
            "full_text": generated_text,
            "tokens": all_tokens,
            "token_ids": all_token_ids,
            "input_length": input_length,
            "attention_weights": attention_weights,
            "token_probabilities": token_probs,
            "generated_ids": generated_ids[input_length:].cpu().numpy()
        }
    
    def visualize_attention(self, tokens: List[str], attention_weights: np.ndarray,
                           layer_idx: int = 0, head_idx: int = 0) -> go.Figure:
        """
        Create attention heatmap visualization.
        
        Args:
            tokens: List of tokens
            attention_weights: Attention weight matrix
            layer_idx: Layer index to visualize
            head_idx: Attention head index to visualize
            
        Returns:
            Plotly figure object
        """
        # Select specific head and layer
        if len(attention_weights.shape) > 3:
            attn = attention_weights[layer_idx, head_idx]
        else:
            attn = attention_weights[head_idx]
        
        # Truncate if needed
        max_len = min(len(tokens), attn.shape[0], attn.shape[1])
        attn = attn[:max_len, :max_len]
        tokens = tokens[:max_len]
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=attn,
            x=tokens,
            y=tokens,
            colorscale='Blues',
            text=np.round(attn, 3),
            texttemplate="%{text}",
            textfont={"size": 8},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title=f"Attention Weights - Layer {layer_idx}, Head {head_idx}",
            xaxis_title="Keys (Attending To)",
            yaxis_title="Queries (Attending From)",
            width=600,
            height=500
        )
        
        return fig
    
    def show_token_probs(self, token_probs: List[np.ndarray], 
                        generated_tokens: List[str], 
                        top_k: int = 10) -> go.Figure:
        """
        Visualize top-k token probabilities for each generation step.
        
        Args:
            token_probs: List of probability distributions
            generated_tokens: List of actually generated tokens
            top_k: Number of top tokens to show
            
        Returns:
            Plotly figure object
        """
        if not token_probs or not generated_tokens:
            return go.Figure()
        
        # Create subplot for each generation step
        n_steps = min(len(token_probs), len(generated_tokens), 5)  # Limit to 5 steps
        fig = make_subplots(
            rows=1, cols=n_steps,
            subplot_titles=[f"Step {i+1}: '{generated_tokens[i]}'" for i in range(n_steps)],
            horizontal_spacing=0.02
        )
        
        colors = px.colors.qualitative.Set3
        
        for step in range(n_steps):
            probs = token_probs[step]
            
            # Get top-k tokens
            top_indices = np.argsort(probs)[-top_k:][::-1]
            top_probs = probs[top_indices]
            top_tokens = [self.tokenizer.decode([idx]) for idx in top_indices]
            
            # Highlight the actually selected token
            bar_colors = ['red' if idx == self.tokenizer.encode(generated_tokens[step])[0] 
                         else colors[i % len(colors)] for i, idx in enumerate(top_indices)]
            
            fig.add_trace(
                go.Bar(
                    x=top_tokens,
                    y=top_probs,
                    name=f"Step {step+1}",
                    marker_color=bar_colors,
                    showlegend=False
                ),
                row=1, col=step+1
            )
        
        fig.update_layout(
            title="Top-K Token Probabilities at Each Generation Step",
            height=400,
            showlegend=False
        )
        
        fig.update_xaxes(tickangle=45)
        
        return fig

def main():
    """Main Streamlit application"""
    
    st.title("🤖 Explainable Transformer Chatbot")
    st.markdown("---")
    
    # Initialize session state
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Sidebar - Model Configuration
    with st.sidebar:
        st.header("🔧 Model Configuration")
        
        # Model selection
        model_options = ["distilgpt2", "gpt2", "gpt2-medium", "microsoft/DialoGPT-small"]
        selected_model = st.selectbox("Select Model", model_options)
        
        # Initialize/Change model
        if st.button("Load Model") or st.session_state.chatbot is None:
            with st.spinner(f"Loading {selected_model}..."):
                st.session_state.chatbot = ExplainableTransformerChatbot(selected_model)
            st.success(f"✅ {selected_model} loaded successfully!")
        
        st.markdown("---")
        
        # Generation parameters
        st.header("⚙️ Generation Parameters")
        temperature = st.slider("Temperature", 0.1, 2.0, 0.7, 0.1)
        top_k = st.slider("Top-K", 1, 100, 50, 1)
        top_p = st.slider("Top-P", 0.1, 1.0, 0.95, 0.05)
        max_length = st.slider("Max Length", 20, 200, 100, 10)
        
        st.markdown("---")
        
        # Fine-tuning section
        st.header("🎯 Fine-tuning")
        uploaded_file = st.file_uploader("Upload training data (.txt)", type="txt")
        
        if uploaded_file is not None:
            training_data = str(uploaded_file.read(), "utf-8")
            
            col1, col2 = st.columns(2)
            with col1:
                epochs = st.number_input("Epochs", 1, 10, 3)
            with col2:
                lr = st.selectbox("Learning Rate", [1e-5, 5e-5, 1e-4, 5e-4], index=1)
            
            if st.button("Start Fine-tuning"):
                if st.session_state.chatbot:
                    with st.spinner("Fine-tuning model..."):
                        success = st.session_state.chatbot.fine_tune_model(
                            training_data, num_epochs=epochs, learning_rate=lr
                        )
                    if success:
                        st.success("✅ Fine-tuning completed!")
                    else:
                        st.error("❌ Fine-tuning failed!")
    
    # Main content area
    if st.session_state.chatbot is None:
        st.warning("Please load a model from the sidebar first.")
        return
    
    # Chat interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Chat Interface")
        
        # User input
        user_input = st.text_area("Your message:", height=100, 
                                 placeholder="Ask me anything...")
        
        if st.button("Generate Response", type="primary"):
            if user_input.strip():
                with st.spinner("Generating response..."):
                    # Generate response
                    result = st.session_state.chatbot.generate_response(
                        user_input, temperature, top_k, top_p, max_length
                    )
                
                # Display response
                st.markdown("### 🤖 Chatbot Response:")
                st.write(result["response"])
                
                # Store in chat history
                st.session_state.chat_history.append({
                    "user": user_input,
                    "bot": result["response"],
                    "analysis": result
                })
                
                # Token Analysis
                st.markdown("### 🔍 Token Analysis")
                token_df = pd.DataFrame({
                    "Token": result["tokens"],
                    "Token ID": result["token_ids"],
                    "Type": ["Input" if i < result["input_length"] else "Generated" 
                            for i in range(len(result["tokens"]))]
                })
                st.dataframe(token_df, use_container_width=True)
                
                # Attention Visualization
                if result["attention_weights"] is not None:
                    st.markdown("### 🎯 Attention Visualization")
                    
                    # Controls for attention visualization
                    col_a, col_b = st.columns(2)
                    with col_a:
                        layer_idx = st.selectbox("Layer", range(len(result["attention_weights"])))
                    with col_b:
                        head_idx = st.selectbox("Head", range(result["attention_weights"][layer_idx].shape[0]))
                    
                    # Generate attention heatmap
                    attention_fig = st.session_state.chatbot.visualize_attention(
                        result["tokens"], result["attention_weights"], layer_idx, head_idx
                    )
                    st.plotly_chart(attention_fig, use_container_width=True)
                
                # Token Probabilities
                if result["token_probabilities"]:
                    st.markdown("### 📊 Token Probabilities")
                    generated_tokens = [result["tokens"][i] for i in range(result["input_length"], 
                                                                         min(result["input_length"] + 5, 
                                                                             len(result["tokens"])))]
                    prob_fig = st.session_state.chatbot.show_token_probs(
                        result["token_probabilities"], generated_tokens
                    )
                    st.plotly_chart(prob_fig, use_container_width=True)
            else:
                st.warning("Please enter a message.")
    
    with col2:
        st.header("📚 Chat History")
        
        # Display chat history
        if st.session_state.chat_history:
            for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):
                with st.expander(f"Chat {len(st.session_state.chat_history)-i}"):
                    st.write(f"**User:** {chat['user']}")
                    st.write(f"**Bot:** {chat['bot']}")
        else:
            st.info("No chat history yet. Start a conversation!")
        
        # Clear history button
        if st.button("Clear History"):
            st.session_state.chat_history = []
            st.experimental_rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **About this app:** This Explainable Transformer Chatbot provides insights into how transformer 
    models generate text through attention visualizations and token probability analysis. 
    You can also fine-tune models on custom datasets.
    """)

if __name__ == "__main__":
    main()
