import React from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, HashRouter } from "react-router-dom";

import App from "./App";
import { STATIC_MODE } from "./lib/api";
import "./theme.css";

/* A static host has no server to rewrite /forecasts back to index.html, so a
   deep link or a refresh on any route but "/" would 404. Hash routing needs no
   rewrite rules at all and works identically on GitHub Pages, Netlify,
   Cloudflare Pages and a file:// copy on a laptop — which is the point, since
   the people opening this link are not going to configure a host. The live
   server keeps clean BrowserRouter URLs. */
const Router = STATIC_MODE ? HashRouter : BrowserRouter;

createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <Router>
      <App />
    </Router>
  </React.StrictMode>
);
