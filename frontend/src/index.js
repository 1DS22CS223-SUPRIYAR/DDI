import React from 'react';
import ReactDOM from 'react-dom/client';
import Home from './components/Home';
import './styles/App.css';
import About from './components/About';
import NavBar from './components/NavBar';
import { Routes, Route, BrowserRouter as Router } from 'react-router-dom';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <Router>
      <NavBar />
      <Routes>
        <Route path='/' element={<Home />} />
        <Route path='/about' element={<About />} />
      </Routes>
    </Router>
  </React.StrictMode>
);