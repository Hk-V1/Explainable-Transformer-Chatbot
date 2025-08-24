import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import google.generativeai as genai
import json
import os
import re
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION - EDIT THESE VALUES
# =============================================================================
GEMINI_API_KEY = "AIzaSyBKCu5YLW9R9WJcVz5QEaHFRXhvd2APbI4"  # Replace with your actual API key
GEMINI_MODEL = "gemini-1.5-flash"  # Options: "gemini-pro", "gemini-pro-vision"
# =============================================================================

# Set up page config
st.set_page_config(
    page_title="Explainable Transformer Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

class GeminiExplainableChatbot:
    """
    A Gemini-powered chatbot with explainability features including
    token analysis and response variations.
    """
    
    def __init__(self, api_key: str, model_name: str = "gemini-pro"):
        """
        Initialize the chatbot with Gemini API.
        
        Args:
            api_key: Google AI API key
            model_name: Gemini model to use
        """
        self.api_key = api_key
        self.model_name = model_name
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        
        # Test connection
        self.is_connected = self._test_connection()
    
    def _test_connection(self) -> bool:
        """Test if the API connection works."""
        try:
            response = self.model.generate_content("Test")
            return True
        except Exception as e:
            st.error(f"Failed to connect to Gemini API: {str(e)}")
            return False
    
    def generate_response(self, prompt: str, temperature: float = 0.7,
                         max_tokens: int = 100, variations: int = 1) -> Dict:
        """
        Generate response using Gemini API with analysis.
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            variations: Number of response variations
            
        Returns:
            Dict containing response and analysis data
        """
        if not self.is_connected:
            return {"error": "Not connected to Gemini API"}
        
        try:
            # Configure generation
            generation_config = genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                candidate_count=1,
            )
            
            # Generate primary response
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            main_response = response.text if response.text else "No response generated"
            
            # Generate variations for analysis
            variations_list = []
            for i in range(min(variations, 3)):  # Limit to 3 variations
                try:
                    var_config = genai.types.GenerationConfig(
                        temperature=min(temperature + 0.2 * i, 1.0),
                        max_output_tokens=max_tokens,
                        candidate_count=1,
                    )
                    
                    var_response = self.model.generate_content(
                        prompt,
                        generation_config=var_config
                    )
                    
                    if var_response.text and var_response.text != main_response:
                        variations_list.append(var_response.text)
                        
                except Exception:
                    continue
            
            # Analyze tokens (simple word-level analysis)
            tokens = self._tokenize_simple(main_response)
            
            # Analyze response characteristics
            analysis = self._analyze_response(main_response, prompt)
            
            return {
                "response": main_response,
                "variations": variations_list,
                "tokens": tokens,
                "token_count": len(tokens),
                "analysis": analysis,
                "prompt_length": len(prompt.split()),
                "response_length": len(main_response.split()),
            }
            
        except Exception as e:
            return {"error": f"Generation failed: {str(e)}"}
    
    def _tokenize_simple(self, text: str) -> List[str]:
        """Simple tokenization for analysis."""
        # Basic word tokenization with punctuation handling
        tokens = re.findall(r'\w+|[^\w\s]', text)
        return tokens
    
    def _analyze_response(self, response: str, prompt: str) -> Dict:
        """Analyze response characteristics."""
        words = response.split()
        
        # Basic metrics
        analysis = {
            "word_count": len(words),
            "char_count": len(response),
            "avg_word_length": np.mean([len(word) for word in words]) if words else 0,
            "sentence_count": len([s for s in response.split('.') if s.strip()]),
            "question_count": response.count('?'),
            "exclamation_count": response.count('!'),
        }
        
        # Sentiment analysis (simple)
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'best']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'worst', 'horrible', 'sad', 'angry']
        
        response_lower = response.lower()
        positive_count = sum(1 for word in positive_words if word in response_lower)
        negative_count = sum(1 for word in negative_words if word in response_lower)
        
        analysis['sentiment_score'] = positive_count - negative_count
        analysis['sentiment_label'] = (
            'Positive' if analysis['sentiment_score'] > 0 else 
            'Negative' if analysis['sentiment_score'] < 0 else 'Neutral'
        )
        
        return analysis
    
    def get_detailed_analysis(self, prompt: str, response: str) -> Dict:
        """Get detailed analysis using Gemini's reasoning capabilities."""
        analysis_prompt = f"""
        Analyze the following conversation and provide insights:
        
        User: {prompt}
        AI: {response}
        
        Please provide analysis in the following format:
        1. Response Quality (1-10):
        2. Relevance to prompt (1-10):
        3. Key themes mentioned:
        4. Tone of response:
        5. Potential improvements:
        
        Keep your analysis concise and structured.
        """
        
        try:
            analysis_response = self.model.generate_content(analysis_prompt)
            return {"detailed_analysis": analysis_response.text}
        except Exception as e:
            return {"detailed_analysis": f"Analysis failed: {str(e)}"}

def create_token_visualization(tokens: List[str]) -> go.Figure:
    """Create a visualization of tokens."""
    if not tokens:
        return go.Figure()
    
    # Token length analysis
    token_lengths = [len(token) for token in tokens]
    
    fig = go.Figure()
    
    # Add bar chart of token lengths
    fig.add_trace(go.Bar(
        x=list(range(len(tokens))),
        y=token_lengths,
        text=tokens,
        textposition='auto',
        hovertemplate='Token: %{text}<br>Length: %{y}<br>Position: %{x}',
        marker_color='lightblue'
    ))
    
    fig.update_layout(
        title="Token Length Analysis",
        xaxis_title="Token Position",
        yaxis_title="Token Length (characters)",
        height=400
    )
    
    return fig

def create_response_metrics_chart(analysis: Dict) -> go.Figure:
    """Create visualization of response metrics."""
    metrics = ['word_count', 'sentence_count', 'question_count', 'exclamation_count']
    values = [analysis.get(metric, 0) for metric in metrics]
    labels = ['Words', 'Sentences', 'Questions', 'Exclamations']
    
    fig = go.Figure(data=[
        go.Bar(
            x=labels,
            y=values,
            text=values,
            textposition='auto',
            marker_color=['#FF9999', '#66B2FF', '#99FF99', '#FFCC99']
        )
    ])
    
    fig.update_layout(
        title="Response Characteristics",
        xaxis_title="Metric",
        yaxis_title="Count",
        height=400
    )
    
    return fig

def create_comparison_chart(responses: List[str]) -> go.Figure:
    """Create comparison chart for response variations."""
    if not responses or len(responses) < 2:
        return go.Figure()
    
    # Analyze each response
    metrics = []
    for i, response in enumerate(responses):
        words = response.split()
        metrics.append({
            'Response': f'Variation {i+1}',
            'Word Count': len(words),
            'Character Count': len(response),
            'Avg Word Length': np.mean([len(word) for word in words]) if words else 0
        })
    
    df = pd.DataFrame(metrics)
    
    fig = go.Figure()
    
    # Add traces for each metric
    fig.add_trace(go.Scatter(
        x=df['Response'],
        y=df['Word Count'],
        mode='lines+markers',
        name='Word Count',
        line=dict(color='blue', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=df['Response'],
        y=df['Character Count'],
        mode='lines+markers',
        name='Character Count',
        yaxis='y2',
        line=dict(color='red', width=3)
    ))
    
    fig.update_layout(
        title="Response Variations Comparison",
        xaxis_title="Response Variation",
        yaxis_title="Word Count",
        yaxis2=dict(title="Character Count", overlaying='y', side='right'),
        height=400
    )
    
    return fig

def main():
    """Main Streamlit application"""
    
    st.title("🤖 Explainable Transformer Chatbot")
    st.markdown("*Powered by Google Gemini API*")
    st.markdown("---")
    
    # Check if API key is configured
    if GEMINI_API_KEY == "YOUR_API_KEY_HERE":
        st.error("🔑 Please configure your API key in the code!")
        st.code('GEMINI_API_KEY = "your-actual-api-key-here"')
        st.markdown("Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey)")
        return
    
    # Initialize session state
    if "chatbot" not in st.session_state:
        with st.spinner(f"Initializing {GEMINI_MODEL}..."):
            st.session_state.chatbot = GeminiExplainableChatbot(GEMINI_API_KEY, GEMINI_MODEL)
            if st.session_state.chatbot.is_connected:
                st.success(f"✅ Connected to {GEMINI_MODEL}!")
            else:
                st.error(f"❌ Failed to connect to {GEMINI_MODEL}")
                return
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Sidebar - Generation Parameters
    with st.sidebar:
        st.header("🤖 Model Info")
        st.info(f"**Model:** {GEMINI_MODEL}")
        st.success("✅ Connected to Gemini API")
        
        st.markdown("---")
        
        # Generation parameters
        st.header("⚙️ Generation Parameters")
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1,
                               help="Higher = more creative, Lower = more focused")
        max_tokens = st.slider("Max Tokens", 50, 500, 150, 25,
                              help="Maximum length of response")
        show_variations = st.checkbox("Show Response Variations", value=True,
                                    help="Generate multiple response variations for comparison")
        
        st.markdown("---")
        
        # Example prompts
        st.header("💡 Example Prompts")
        example_prompts = [
            "Explain quantum computing in simple terms",
            "Write a short story about AI and humans",
            "What are the benefits of renewable energy?",
            "How does machine learning work?",
            "Describe the future of space exploration"
        ]
        
        for prompt in example_prompts:
            if st.button(f"'{prompt[:25]}...'", key=prompt):
                st.session_state.example_prompt = prompt
    
    # Main content area
    if st.session_state.chatbot is None or not st.session_state.chatbot.is_connected:
        st.warning("⚠️ Failed to initialize Gemini API. Please check your API key.")
        return
    
    # Chat interface
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.header("💬 Chat Interface")
        
        # User input
        default_prompt = getattr(st.session_state, 'example_prompt', '')
        user_input = st.text_area("Your prompt:", height=100, value=default_prompt,
                                 placeholder="Ask me anything...")
        
        # Clear example prompt after use
        if hasattr(st.session_state, 'example_prompt'):
            delattr(st.session_state, 'example_prompt')
        
        col_gen, col_analyze = st.columns(2)
        
        with col_gen:
            generate_btn = st.button("🎯 Generate Response", type="primary")
        
        with col_analyze:
            analyze_btn = st.button("🔍 Deep Analysis")
        
        if generate_btn and user_input.strip():
            with st.spinner("🤖 Generating response..."):
                # Generate response
                variations_count = 3 if show_variations else 1
                result = st.session_state.chatbot.generate_response(
                    user_input, temperature, max_tokens, variations_count
                )
            
            if "error" in result:
                st.error(result["error"])
            else:
                # Display main response
                st.markdown("### 🤖 Generated Response:")
                st.success(result["response"])
                
                # Store in chat history
                st.session_state.chat_history.append({
                    "user": user_input,
                    "bot": result["response"],
                    "analysis": result
                })
                
                # Response Analysis
                st.markdown("### 📊 Response Analysis")
                
                col_metrics, col_tokens = st.columns(2)
                
                with col_metrics:
                    # Display key metrics
                    analysis = result.get("analysis", {})
                    st.metric("Words Generated", analysis.get("word_count", 0))
                    st.metric("Characters", analysis.get("char_count", 0))
                    st.metric("Sentiment", analysis.get("sentiment_label", "Neutral"))
                
                with col_tokens:
                    st.metric("Total Tokens", result.get("token_count", 0))
                    st.metric("Avg Word Length", f"{analysis.get('avg_word_length', 0):.1f}")
                    st.metric("Sentences", analysis.get("sentence_count", 0))
                
                # Token Visualization
                if result.get("tokens"):
                    st.markdown("### 🔤 Token Analysis")
                    
                    # Show tokens as tags
                    st.markdown("**Generated Tokens:**")
                    tokens_html = " ".join([f'<span style="background-color: lightblue; padding: 2px 6px; margin: 2px; border-radius: 3px; display: inline-block;">{token}</span>' for token in result["tokens"][:20]])
                    st.markdown(tokens_html, unsafe_allow_html=True)
                    
                    if len(result["tokens"]) > 20:
                        st.caption(f"... and {len(result['tokens']) - 20} more tokens")
                    
                    # Token length visualization
                    token_fig = create_token_visualization(result["tokens"][:30])
                    st.plotly_chart(token_fig, use_container_width=True)
                
                # Response Characteristics Chart
                if analysis:
                    st.markdown("### 📈 Response Characteristics")
                    metrics_fig = create_response_metrics_chart(analysis)
                    st.plotly_chart(metrics_fig, use_container_width=True)
                
                # Variations Comparison
                if result.get("variations") and show_variations:
                    st.markdown("### 🎲 Response Variations")
                    
                    with st.expander("View All Variations", expanded=True):
                        all_responses = [result["response"]] + result["variations"]
                        
                        for i, variation in enumerate(all_responses):
                            st.markdown(f"**Variation {i+1}:**")
                            st.write(variation)
                            st.markdown("---")
                        
                        # Comparison chart
                        if len(all_responses) > 1:
                            comparison_fig = create_comparison_chart(all_responses)
                            st.plotly_chart(comparison_fig, use_container_width=True)
        
        if analyze_btn and user_input.strip():
            if st.session_state.chat_history:
                last_chat = st.session_state.chat_history[-1]
                
                with st.spinner("🔍 Performing deep analysis..."):
                    deep_analysis = st.session_state.chatbot.get_detailed_analysis(
                        last_chat["user"], last_chat["bot"]
                    )
                
                st.markdown("### 🧠 Deep Analysis")
                if "detailed_analysis" in deep_analysis:
                    st.markdown(deep_analysis["detailed_analysis"])
                else:
                    st.error("Deep analysis failed")
            else:
                st.warning("Generate a response first to analyze it.")
    
    with col2:
        st.header("📚 Chat History")
        
        # Display recent chat history
        if st.session_state.chat_history:
            for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):
                with st.expander(f"💬 Chat {len(st.session_state.chat_history)-i}", expanded=(i==0)):
                    st.markdown(f"**🧑 You:** {chat['user']}")
                    st.markdown(f"**🤖 Bot:** {chat['bot']}")
                    
                    # Show quick stats
                    if 'analysis' in chat and 'analysis' in chat['analysis']:
                        analysis = chat['analysis']['analysis']
                        st.caption(f"📊 {analysis.get('word_count', 0)} words, "
                                 f"{analysis.get('sentiment_label', 'Neutral')} sentiment")
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
                chat_data = []
                for chat in st.session_state.chat_history:
                    chat_data.append({
                        "user_input": chat["user"],
                        "bot_response": chat["bot"],
                        "timestamp": "now"  # You can add proper timestamps
                    })
                
                st.download_button(
                    "💾 Download",
                    data=json.dumps(chat_data, indent=2),
                    file_name="gemini_chat_history.json",
                    mime="application/json"
                )
    
    # Footer with information
    st.markdown("---")
    with st.expander("ℹ️ About this Application"):
        st.markdown("""
        ### 🔬 Explainable Transformer Chatbot (Gemini-Powered)
        
        This application demonstrates how AI language models work by providing:
        
        **🎯 Core Features:**
        - **Gemini Integration**: Powered by Google's advanced AI
        - **Response Analysis**: Token breakdown and characteristics
        - **Variations**: Multiple response options for comparison  
        - **Deep Analysis**: AI-powered analysis of conversations
        - **Interactive Visualizations**: Charts and metrics
        
        **🧠 Understanding the Analysis:**
        - **Tokens**: Individual pieces of text the AI processes
        - **Sentiment**: Emotional tone of the response
        - **Variations**: Different possible responses showing AI creativity
        - **Metrics**: Statistical analysis of response characteristics
        
        **⚙️ Parameters:**
        - **Temperature**: Controls creativity (0.0 = focused, 1.0 = creative)
        - **Max Tokens**: Maximum length of response
        
        **🔑 API Key Required:**
        Get your free API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
        
        Built with Streamlit, Google Gemini API, and Plotly.
        """)

if __name__ == "__main__":
    main()
