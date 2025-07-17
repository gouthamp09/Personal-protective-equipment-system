// import React from 'react';
// import ReactDOM from 'react-dom/client';
// import './index.css';
// import App from './App';
// import reportWebVitals from './reportWebVitals';
// import { GoogleOAuthProvider } from '@react-oauth/google';
// const root = ReactDOM.createRoot(document.getElementById('root'));
// root.render(
  
//   <React.StrictMode>
//     <GoogleOAuthProvider clientId="233108575371-77e0npscvnpec8rmbpqed5j1qa5hk0ls.apps.googleusercontent.com">
//       <App />
//     </GoogleOAuthProvider>
//   </React.StrictMode>
// );

// // If you want to start measuring performance in your app, pass a function
// // to log results (for example: reportWebVitals(console.log))
// // or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
// reportWebVitals();
import React from "react";
import ReactDOM from "react-dom/client";

import App from "./App";
import { GoogleOAuthProvider } from "@react-oauth/google";
import "./App.css";
import Test from "./Test";
const root = ReactDOM.createRoot(document.getElementById("root"));

root.render(
 
    <GoogleOAuthProvider clientId="233108575371-77e0npscvnpec8rmbpqed5j1qa5hk0ls.apps.googleusercontent.com">
      <App/>
    </GoogleOAuthProvider>
  
);