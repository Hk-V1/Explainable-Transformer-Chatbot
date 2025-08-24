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
    pipeline, set_seed
)
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
        
        # Initialize pipeline for simple generation
        self.pipeline = pipeline(
            'text-generation', 
            model=model_name,
            device=0 if torch.cuda.is_available() else -1,
            return_full_text=True
        )
        
        # Load tokenizer and model for detailed analysis
        try:
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
            self.detailed_model_available = True
            
        except Exception as e:
            st.warning(f"Detailed model analysis not available: {e}")
            self.tokenizer = None
            self.model = None
            self.detailed_model_available = False
    
    def simple_generate(self, prompt: str, temperature: float = 0.7,
                       top_k: int = 50, top_p: float = 0.95,
                       max_length: int = 100, seed: int = 42) -> str:
        """
        Simple text generation using pipeline.
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            max_length: Maximum generation length
            seed: Random seed for reproducibility
            
        Returns:
            Generated text
        """
        set_seed(seed)
        
        try:
            # Configure generation parameters
            generation_kwargs = {
                'max_length': max_length,
                'temperature': temperature,
                'top_k': top_k,
                'top_p': top_p,
                'do_sample': True,
                'num_return_sequences': 1,
                'pad_token_id': self.pipeline.tokenizer.eos_token_id
            }
            
            # Generate text
            result = self.pipeline(prompt, **generation_kwargs)
            generated_text = result[0]['generated_text']
            
            # Extract only the new part (response)
            response = generated_text[len(prompt):].strip()
            
            return response
            
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def generate_response(self, prompt: str, temperature: float = 0.7,
                         top_k: int = 50, top_p: float = 0.95,
                         max_length: int = 100, seed: int = 42) -> Dict:
        """
        Generate response with detailed information for explainability.
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            max_length: Maximum generation length
            seed: Random seed for reproducibility
            
        Returns:
            Dict containing response and analysis data
        """
        # First get simple response
        response = self.simple_generate(prompt, temperature, top_k, top_p, max_length, seed)
        
        result = {
            "response": response,
            "full_text": prompt + " " + response,
            "tokens": [],
            "token_ids": [],
            "input_length": 0,
            "attention_weights": None,
            "token_probabilities": [],
            "generated_ids": []
        }
        
        # If detailed model is available, get additional analysis
        if self.detailed_model_available and self.tokenizer and self.model:
            try:
                # Tokenize full text
                full_text = prompt + " " + response
                input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
                full_ids = self.tokenizer.encode(full_text, return_tensors="pt").to(self.device)
                
                # Get tokens and IDs
                all_tokens = self.tokenizer.convert_ids_to_tokens(full_ids[0])
                all_token_ids = full_ids[0].cpu().numpy().tolist()
                input_length = input_ids.shape[1]
                
                # Get model outputs for attention analysis
                with torch.no_grad():
                    outputs = self.model(full_ids, output_attentions=True)
                    
                # Extract attention weights
                attentions = outputs.attentions
                if attentions:
                    # Get last layer attention
                    last_layer_attention = attentions[-1][0].cpu().numpy()
                    result["attention_weights"] = last_layer_attention
                
                # Update result with detailed info
                result.update({
                    "tokens": all_tokens,
                    "token_ids": all_token_ids,
                    "input_length": input_length,
                    "generated_ids": full_ids[0][input_length:].cpu().numpy()
                })
                
            except Exception as e:
                st.warning(f"Detailed analysis failed: {e}")
        
        return result
    
    def get_token_probabilities_simple(self, prompt: str, num_variations: int = 5) -> Dict:
        """
        Get multiple generation variations to show different possible continuations.
        
        Args:
            prompt: Input prompt
            num_variations: Number of different continuations to generate
            
        Returns:
            Dict with variations and their relative frequencies
        """
        try:
            # Generate multiple variations
            results = self.pipeline(
                prompt,
                max_length=len(prompt.split()) + 10,
                num_return_sequences=num_variations,
                temperature=0.8,
                do_sample=True,
                pad_token_id=self.pipeline.tokenizer.eos_token_id
            )
            
            # Extract responses
            responses = []
            for result in results:
                response = result['generated_text'][len(prompt):].strip()
                if response:
                    responses.append(response)
            
            # Count frequencies of first few words
            first_words = {}
            for response in responses:
                words = response.split()
                if words:
                    first_word = words[0]
                    first_words[first_word] = first_words.get(first_word, 0) + 1
            
            return {
                "variations": responses,
                "first_word_counts": first_words
            }
            
        except Exception as e:
            return {"variations": [], "first_word_counts": {}}
    
    def visualize_attention(self, tokens: List[str], attention_weights: np.ndarray,
                           layer_idx: int = -1, head_idx: int = 0) -> go.Figure:
        """
        Create attention heatmap visualization.
        
        Args:
            tokens: List of tokens
            attention_weights: Attention weight matrix
            layer_idx: Layer index to visualize (-1 for last layer)
            head_idx: Attention head index to visualize
            
        Returns:
            Plotly figure object
        """
        if attention_weights is None:
            return go.Figure()
        
        # Select specific head
        if len(attention_weights.shape) >= 3:
            attn = attention_weights[head_idx]
        else:
            attn = attention_weights
        
        # Truncate if needed for visualization
        max_len = min(len(tokens), attn.shape[0], attn.shape[1], 50)
        attn = attn[:max_len, :max_len]
        tokens_display = tokens[:max_len]
        
        # Truncate long tokens for display
        tokens_display = [tok[:8] + "..." if len(tok) > 8 else tok for tok in tokens_display]
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=attn,
            x=tokens_display,
            y=tokens_display,
            colorscale='Blues',
            text=np.round(attn, 3),
            texttemplate="%{text}",
            textfont={"size": 8},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title=f"Attention Weights - Head {head_idx}",
            xaxis_title="Keys (Attending To)",
            yaxis_title="Queries (Attending From)",
            width=600,
            height=500,
            xaxis={'side': 'bottom'},
            yaxis={'side': 'left'}
        )
        
        return fig
    
    def show_generation_variations(self, variations_data: Dict) -> go.Figure:
        """
        Visualize generation variations and their frequencies.
        
        Args:
            variations_data: Dictionary with variations and word counts
            
        Returns:
            Plotly figure object
        """
        if not variations_data["first_word_counts"]:
            return go.Figure()
        
        words = list(variations_data["first_word_counts"].keys())
        counts = list(variations_data["first_word_counts"].values())
        
        fig = go.Figure(data=[
            go.Bar(
                x=words,
                y=counts,
                text=counts,
                textposition='auto',
                marker_color='lightblue'
            )
        ])
        
        fig.update_layout(
            title="First Word Frequency in Generated Variations",
            xaxis_title="First Word",
            yaxis_title="Frequency",
            height=400
        )
        
        return fig

def main():
    """Main Streamlit application"""
    
    st.title("🤖 Explainable Transformer Chatbot")
    st.markdown("*Understanding how language models generate text*")
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
        model_options = [
            "distilgpt2", 
            "gpt2", 
            "gpt2-medium",
            "microsoft/DialoGPT-small",
            "microsoft/DialoGPT-medium"
        ]
        selected_model = st.selectbox("Select Model", model_options)
        
        # Initialize/Change model
        if st.button("Load Model") or st.session_state.chatbot is None:
            with st.spinner(f"Loading {selected_model}..."):
                try:
                    st.session_state.chatbot = ExplainableTransformerChatbot(selected_model)
                    st.success(f"✅ {selected_model} loaded successfully!")
                except Exception as e:
                    st.error(f"Failed to load model: {e}")
                    st.session_state.chatbot = None
        
        if st.session_state.chatbot:
            # Show model status
            if st.session_state.chatbot.detailed_model_available:
                st.success("🔬 Detailed analysis available")
            else:
                st.info("📝 Basic generation available")
        
        st.markdown("---")
        
        # Generation parameters
        st.header("⚙️ Generation Parameters")
        temperature = st.slider("Temperature", 0.1, 2.0, 0.7, 0.1, 
                               help="Higher = more creative, Lower = more focused")
        top_k = st.slider("Top-K", 1, 100, 50, 1,
                         help="Consider only top K tokens")
        top_p = st.slider("Top-P", 0.1, 1.0, 0.95, 0.05,
                         help="Nucleus sampling threshold")
        max_length = st.slider("Max Length", 20, 200, 100, 10,
                              help="Maximum tokens to generate")
        seed = st.number_input("Random Seed", 0, 1000, 42,
                              help="For reproducible results")
        
        st.markdown("---")
        
        # Example prompts
        st.header("💡 Example Prompts")
        example_prompts = [
            "The future of artificial intelligence is",
            "In a world where robots and humans coexist",
            "The most important invention in history was",
            "Climate change will affect our planet by",
            "The secret to happiness is"
        ]
        
        for prompt in example_prompts:
            if st.button(f"'{prompt[:30]}...'", key=prompt):
                st.session_state.example_prompt = prompt
    
    # Main content area
    if st.session_state.chatbot is None:
        st.warning("⚠️ Please load a model from the sidebar first.")
        return
    
    # Chat interface
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.header("💬 Chat Interface")
        
        # User input
        default_prompt = getattr(st.session_state, 'example_prompt', '')
        user_input = st.text_area("Your prompt:", height=100, value=default_prompt,
                                 placeholder="Enter your prompt here...")
        
        # Clear example prompt after use
        if hasattr(st.session_state, 'example_prompt'):
            delattr(st.session_state, 'example_prompt')
        
        col_gen, col_var = st.columns(2)
        
        with col_gen:
            generate_response = st.button("🎯 Generate Response", type="primary")
        
        with col_var:
            show_variations = st.button("🎲 Show Variations")
        
        if generate_response and user_input.strip():
            with st.spinner("🤖 Generating response..."):
                # Generate response
                result = st.session_state.chatbot.generate_response(
                    user_input, temperature, top_k, top_p, max_length, seed
                )
            
            # Display response
            st.markdown("### 🤖 Generated Response:")
            st.success(result["response"])
            
            # Store in chat history
            st.session_state.chat_history.append({
                "user": user_input,
                "bot": result["response"],
                "analysis": result
            })
            
            # Token Analysis (if available)
            if result["tokens"]:
                st.markdown("### 🔍 Token Analysis")
                
                # Create token dataframe
                token_df = pd.DataFrame({
                    "Position": range(len(result["tokens"])),
                    "Token": result["tokens"],
                    "Token ID": result["token_ids"],
                    "Type": ["Input" if i < result["input_length"] else "Generated" 
                            for i in range(len(result["tokens"]))]
                })
                
                # Color code the dataframe
                def color_tokens(val):
                    if val == "Input":
                        return 'background-color: lightblue'
                    else:
                        return 'background-color: lightgreen'
                
                styled_df = token_df.style.applymap(color_tokens, subset=['Type'])
                st.dataframe(styled_df, use_container_width=True)
                
                # Attention Visualization (if available)
                if result["attention_weights"] is not None:
                    st.markdown("### 🎯 Attention Visualization")
                    
                    # Controls for attention visualization
                    col_head = st.columns(1)[0]
                    with col_head:
                        max_heads = result["attention_weights"].shape[0] if len(result["attention_weights"].shape) > 2 else 1
                        head_idx = st.selectbox("Attention Head", range(max_heads))
                    
                    # Generate attention heatmap
                    attention_fig = st.session_state.chatbot.visualize_attention(
                        result["tokens"], result["attention_weights"], head_idx=head_idx
                    )
                    st.plotly_chart(attention_fig, use_container_width=True)
                    
                    st.info("💡 **How to read:** Darker colors show stronger attention. "
                           "Each row shows what tokens that position attends to.")
        
        if show_variations and user_input.strip():
            with st.spinner("🎲 Generating variations..."):
                variations = st.session_state.chatbot.get_token_probabilities_simple(user_input)
            
            st.markdown("### 🎲 Generation Variations")
            st.info("Multiple possible continuations to show model uncertainty:")
            
            for i, variation in enumerate(variations["variations"], 1):
                st.write(f"**{i}.** {variation}")
            
            # Show frequency chart
            if variations["first_word_counts"]:
                st.markdown("### 📊 First Word Frequency")
                freq_fig = st.session_state.chatbot.show_generation_variations(variations)
                st.plotly_chart(freq_fig, use_container_width=True)
    
    with col2:
        st.header("📚 Chat History")
        
        # Display recent chat history
        if st.session_state.chat_history:
            for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):
                with st.expander(f"💬 Chat {len(st.session_state.chat_history)-i}", expanded=(i==0)):
                    st.markdown(f"**🧑 You:** {chat['user']}")
                    st.markdown(f"**🤖 Bot:** {chat['bot']}")
                    
                    # Show quick stats if available
                    if chat['analysis']['tokens']:
                        n_input = chat['analysis']['input_length']
                        n_generated = len(chat['analysis']['tokens']) - n_input
                        st.caption(f"📊 {n_input} input tokens → {n_generated} generated tokens")
        else:
            st.info("💭 No chat history yet. Start a conversation!")
        
        # Control buttons
        col_clear, col_download = st.columns(2)
        with col_clear:
            if st.button("🗑️ Clear History"):
                st.session_state.chat_history = []
                st.rerun()
        
        with col_download:
            if st.session_state.chat_history:
                # Prepare download data
                chat_data = []
                for chat in st.session_state.chat_history:
                    chat_data.append({
                        "user_input": chat["user"],
                        "bot_response": chat["bot"]
                    })
                
                st.download_button(
                    "💾 Download",
                    data=json.dumps(chat_data, indent=2),
                    file_name="chat_history.json",
                    mime="application/json"
                )
    
    # Footer with information
    st.markdown("---")
    with st.expander("ℹ️ About this Application"):
        st.markdown("""
        ### 🔬 Explainable Transformer Chatbot
        
        This application demonstrates how transformer language models work by providing:
        
        **🎯 Core Features:**
        - **Text Generation**: State-of-the-art transformer models
        - **Token Analysis**: See how text is broken into tokens
        - **Attention Visualization**: Understand what the model focuses on
        - **Generation Variations**: Explore different possible outputs
        
        **🧠 Understanding the Visualizations:**
        - **Tokens**: Words/subwords the model processes
        - **Attention**: What parts of the input the model considers when generating each word
        - **Variations**: Different possible continuations showing model uncertainty
        
        **⚙️ Parameters:**
        - **Temperature**: Controls creativity (higher = more random)
        - **Top-K**: Only consider the K most likely next tokens
        - **Top-P**: Consider tokens until cumulative probability reaches P
        - **Max Length**: Maximum number of tokens to generate
        
        Built with Streamlit, Transformers, and Plotly for interactive AI explanation.
        """)

if __name__ == "__main__":
    main()
