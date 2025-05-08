import React from "react";
import { useNavigate } from "react-router-dom";

function NavBar() {
  const navigate = useNavigate();
  
  return (
    <nav className="top-nav">
      <div className="app-name">Drugram</div>
      <div className="nav-links">
        <button 
          className="nav-link" 
          onClick={() => navigate('/')}
        >
          Home
        </button>
        <button 
          className="nav-link" 
          onClick={() => navigate('/about')}
        >
          About
        </button>
      </div>
    </nav>
  );
}

export default NavBar;