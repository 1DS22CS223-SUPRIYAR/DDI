import React from 'react';
import '../styles/About.css';

const About = () => {
  return (
    <div className="about-container">
      <div className="about-header">
        <h1>About Drugram</h1>
        <p className="subtitle">Predicting Drug-Drug Interactions through Molecular Graph Neural Networks</p>
      </div>

      <div className="about-content">
        <section className="about-section">
          <h2>Our Innovation</h2>
          <p className="highlight-text">
            Drugram revolutionizes drug interaction prediction by analyzing the 3D spatial structure 
            of molecules using advanced graph neural networks, uncovering interactions that traditional 
            methods often miss.
          </p>
        </section>

        <section className="about-section">
          <h2>Who Benefits</h2>
          <div className="user-roles">
            <div className="role-card">
              <div className="icon-wrapper">🧪</div>
              <h3>Pharmaceutical Researchers</h3>
              <p>Screen drug combinations with structural insights during preclinical development</p>
            </div>
            <div className="role-card">
              <div className="icon-wrapper">💻</div>
              <h3>Computational Chemists</h3>
              <p>Integrate our graph-based API into your discovery pipelines</p>
            </div>
            <div className="role-card">
              <div className="icon-wrapper">🔍</div>
              <h3>Medicinal Chemists</h3>
              <p>Visualize how molecular modifications affect interaction profiles</p>
            </div>
          </div>
        </section>

        <section className="about-section workflow-section">
          <h2>How It Works</h2>
          <div className="workflow-steps">
            <div className="workflow-card">
              <div className="step-content">
                <h3>1. Input Molecules</h3>
                <p>Enter drug names, SMILES strings, or upload molecular structures</p>
                <div className="molecule-visual">CCO.NC(=O)C1=CC=CC=C1</div>
              </div>
            </div>
            
            <div className="workflow-arrow">→</div>
            
            <div className="workflow-card">
        
              <div className="step-content">
                <h3>2. 3D Graph Conversion</h3>
                <p>Our system converts molecules into spatial graph representations</p>
                <div className="graph-visual">[Molecular Graph Icon]</div>
              </div>
            </div>
            
            <div className="workflow-arrow">→</div>
            
            <div className="workflow-card">
             
              <div className="step-content">
                <h3>3. Interaction Analysis</h3>
                <div className="interaction-visual">⚡ + ❌ = ⚠️</div>
              </div>
            </div>
            
            <div className="workflow-arrow">→</div>
            
            
          </div>
        </section>

        <section className="about-section feature-showcase">
          <h2>Key Features</h2>
          <div className="features-grid">
            <div className="feature-card">
              <div className="feature-icon">🧠</div>
              <h3>Deep Learning Engine</h3>
              <p>Graph neural networks trained on 25,000+ known drug interactions</p>
            </div>
            <div className="feature-card">
              <div className="feature-icon">📊</div>
              <h3>Explainable AI</h3>
              <p>Attention maps highlight contributing molecular substructures</p>
            </div>
            <div className="feature-card">
              <div className="feature-icon">🔄</div>
              <h3>Continuous Learning</h3>
              <p>Model improves as new interaction data becomes available</p>
            </div>
          </div>
        </section>

        <section className="about-section cta-section">
          <h2>Ready to Explore?</h2>
          <p>
            Start predicting drug interactions with molecular precision today.
            Our web interface makes interaction prediction seamless.
          </p>
          <button 
            className="cta-button"
            onClick={() => window.location.href = 'root'}
            >
            Get Started
          </button>   
        </section>

        <section className="about-section disclaimer">
          <h2>Important Note</h2>
          <p>
            Predictions require experimental validation. Achieves 89.2% accuracy on DrugBank DDI benchmark.
            Not a substitute for clinical evaluation.
          </p>
        </section>
      </div>
    </div>
  );
};

export default About;