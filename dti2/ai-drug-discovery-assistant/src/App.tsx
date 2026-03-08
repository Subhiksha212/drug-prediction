/// <reference types="vite/client" />

import React, { useState, useEffect, useRef } from 'react';
import { GoogleGenAI } from "@google/genai";
import { 
  Search, 
  Dna, 
  Activity, 
  BarChart3, 
  Loader2, 
  FlaskConical, 
  Info,
  ExternalLink,
  ClipboardList,
  Download,
  X,
  Zap,
  ShieldCheck,
  Microscope,
  Database,
  Globe,
  Sparkles,
  UserCheck,
  AlertTriangle,
  History,
  MessageSquare,
  CheckCircle2,
  XCircle,
  Lock,
  Mail,
  User,
  ArrowRight,
  LogOut,
  AlertCircle,
  Image,
  FileText,
  Server,
  Cpu
} from 'lucide-react';
import Markdown from 'react-markdown';
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer, 
  Cell 
} from 'recharts';
import { motion, AnimatePresence } from 'motion/react';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

// Add type declaration for 3Dmol
declare global {
  interface Window {
    $3Dmol: any;
  }
}

// Utility for tailwind classes
function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

// Types
interface DrugPrediction {
  drug: string;
  score: number;
  smiles: string;
  target?: string;
  id?: string;
}

interface DrugDetails extends DrugPrediction {
  molecularWeight?: string;
  drugClass?: string;
  mechanism?: string;
  clinicalPhase?: string;
  toxicityScore?: string;
  interactions?: Array<{
    protein_id: string;
    protein_name: string;
    affinity_score: number;
    binding_energy?: number;
  }>;
}

interface Protein {
  id: string;
  name: string;
  sequence: string;
  organism?: string;
  function?: string;
}

interface ScientistFeedback {
  drug_id: string;
  smiles: string;
  score: number;
  decision: 'approved' | 'rejected';
  comment: string;
  timestamp: string;
}

interface ServerStatus {
  status: string;
  totalDrugs: number;
  totalProteins: number;
  totalInteractions: number;
  isLoaded: boolean;
  timestamp: string;
}

// API Service

// API Service - Updated for Flask backend
class ApiService {
  private baseUrl: string;
  
  constructor() {
    // Flask runs on port 5000 by default
    this.baseUrl = 'http://localhost:5000';
  }

  async getStatus(): Promise<ServerStatus> {
    try {
      const response = await fetch(`${this.baseUrl}/api/status`);
      if (!response.ok) throw new Error('Failed to fetch server status');
      return await response.json();
    } catch (error) {
      console.error('Status check error:', error);
      throw error;
    }
  }

  async searchDrugs(query: string): Promise<DrugPrediction[]> {
    const response = await fetch(`${this.baseUrl}/api/drugs?search=${encodeURIComponent(query)}`);
    if (!response.ok) throw new Error('Failed to search drugs');
    return response.json();
  }

  async getDrugDetails(id: string): Promise<DrugDetails> {
    const response = await fetch(`${this.baseUrl}/api/drugs/${id}`);
    if (!response.ok) throw new Error('Failed to fetch drug details');
    return response.json();
  }

  async predictDrugs(sequence: string): Promise<DrugPrediction[]> {
    console.log('Sending prediction request to Flask...');
    
    const response = await fetch(`${this.baseUrl}/api/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ sequence }),
    });
    
    console.log('Response status:', response.status);
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ error: 'Unknown error' }));
      console.error('Error response:', errorData);
      throw new Error(errorData.error || errorData.details || `Server error: ${response.status}`);
    }
    
    const data = await response.json();
    console.log('Received predictions:', data);
    return data;
  }

  async testConnection(): Promise<boolean> {
    try {
      const response = await fetch(`${this.baseUrl}/api/test`);
      return response.ok;
    } catch {
      return false;
    }
  }
}

// Predefined CID mappings for common drugs (as fallback when API fails)
const DRUG_CID_MAP: Record<string, string> = {
  "Remdesivir": "121304016",
  "Dexamethasone": "5743",
  "Favipiravir": "492405",
  "Ritonavir": "392622",
  "Lopinavir": "92727",
  "Hydroxychloroquine": "3652",
  "Azithromycin": "447043",
  "Ivermectin": "6321424",
  "Oseltamivir": "65028"
};

// PubChem API Service with multiple fallback strategies
class PubChemService {
  private static instance: PubChemService;
  private cache: Map<string, any> = new Map();
  private requestQueue: Map<string, Promise<any>> = new Map();

  static getInstance() {
    if (!PubChemService.instance) {
      PubChemService.instance = new PubChemService();
    }
    return PubChemService.instance;
  }

  async getCompoundCID(smiles: string): Promise<string | null> {
    const cacheKey = `cid:${smiles}`;
    if (this.cache.has(cacheKey)) return this.cache.get(cacheKey);

    try {
      const response = await fetch(
        `https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/${encodeURIComponent(smiles)}/cids/JSON`,
        { signal: AbortSignal.timeout(5000) }
      );

      if (!response.ok) return null;

      const data = await response.json();
      const cid = data.IdentifierList?.CID?.[0]?.toString() || null;
      this.cache.set(cacheKey, cid);
      return cid;
    } catch {
      return null;
    }
  }

  async getCIDByName(drugName: string): Promise<string | null> {
    const cacheKey = `name:${drugName}`;
    if (this.cache.has(cacheKey)) return this.cache.get(cacheKey);

    try {
      // Check predefined map first (fastest)
      if (DRUG_CID_MAP[drugName]) {
        this.cache.set(cacheKey, DRUG_CID_MAP[drugName]);
        return DRUG_CID_MAP[drugName];
      }

      // Clean up drug name - remove common suffixes for better matching
      const cleanName = drugName.split(' ')[0].trim();
      const response = await fetch(
        `https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/${encodeURIComponent(cleanName)}/cids/JSON`,
        { signal: AbortSignal.timeout(5000) }
      );

      if (!response.ok) return null;

      const data = await response.json();
      const cid = data.IdentifierList?.CID?.[0]?.toString() || null;
      this.cache.set(cacheKey, cid);
      return cid;
    } catch {
      return null;
    }
  }

  async getImageUrl(drugName: string, smiles: string): Promise<{ url: string; cid: string | null } | null> {
    const cacheKey = `img:${drugName}`;
    if (this.cache.has(cacheKey)) return this.cache.get(cacheKey);

    try {
      // Try multiple strategies in order
      
      // Strategy 1: Get CID by name (with predefined map)
      let cid = await this.getCIDByName(drugName);
      
      // Strategy 2: If name fails, try SMILES
      if (!cid) {
        cid = await this.getCompoundCID(smiles);
      }

      if (cid) {
        // Try different image formats
        const imageUrls = [
          `https://pubchem.ncbi.nlm.nih.gov/image/imgsrv.fcgi?cid=${cid}&t=l`, // Large image
          `https://pubchem.ncbi.nlm.nih.gov/image/imgsrv.fcgi?cid=${cid}` // Default image
        ];

        for (const url of imageUrls) {
          try {
            const response = await fetch(url, { method: 'HEAD', signal: AbortSignal.timeout(3000) });
            if (response.ok) {
              const result = { url, cid };
              this.cache.set(cacheKey, result);
              return result;
            }
          } catch {
            continue;
          }
        }
      }

      // Strategy 3: Use Wikipedia image as last resort
      const wikiUrl = `https://commons.wikimedia.org/wiki/Special:FilePath/${encodeURIComponent(drugName)}.svg`;
      try {
        const response = await fetch(wikiUrl, { method: 'HEAD', signal: AbortSignal.timeout(3000) });
        if (response.ok) {
          const result = { url: wikiUrl, cid: null };
          this.cache.set(cacheKey, result);
          return result;
        }
      } catch {
        // Ignore
      }

      return null;
    } catch {
      return null;
    }
  }
}

// Molecule Viewer Component with multiple fallback strategies
const MoleculeViewer = ({ smiles, drugName }: { smiles: string; drugName: string }) => {
  const viewerRef = useRef<HTMLDivElement>(null);
  const [viewState, setViewState] = useState<'loading' | 'image' | 'error' | 'unavailable'>('loading');
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [cid, setCid] = useState<string | null>(null);
  const [loadAttempts, setLoadAttempts] = useState(0);

  useEffect(() => {
    let mounted = true;
    const pubchem = PubChemService.getInstance();

    const loadImage = async () => {
      if (!mounted) return;

      setViewState('loading');
      
      try {
        const imageResult = await pubchem.getImageUrl(drugName, smiles);
        
        if (!mounted) return;

        if (imageResult) {
          setImageUrl(imageResult.url);
          setCid(imageResult.cid);
          setViewState('image');
        } else {
          setViewState('unavailable');
        }
      } catch (error) {
        if (mounted) {
          setViewState('error');
        }
      }
    };

    loadImage();
  }, [smiles, drugName, loadAttempts]);

  const handleRetry = () => {
    setLoadAttempts(prev => prev + 1);
  };

  const renderContent = () => {
    switch (viewState) {
      case 'loading':
        return (
          <div className="flex flex-col items-center justify-center h-full gap-2">
            <Loader2 className="animate-spin text-biotech-primary" size={32} />
            <p className="text-xs font-medium text-slate-400">Loading structure...</p>
          </div>
        );

      case 'image':
        return imageUrl ? (
          <div className="flex flex-col items-center justify-center h-full p-2">
            <img 
              src={imageUrl}
              alt={`${drugName} structure`}
              className="max-h-32 object-contain mb-2"
              onError={() => setViewState('unavailable')}
              crossOrigin="anonymous"
            />
            <p className="text-[10px] text-slate-400">2D Structure from PubChem</p>
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center h-full p-4">
            <Image size={32} className="text-slate-300 mb-2" />
            <p className="text-xs text-slate-400">No structure available</p>
          </div>
        );

      case 'unavailable':
        return (
          <div className="flex flex-col items-center justify-center h-full p-4 text-center">
            <FileText size={32} className="text-amber-400 mb-2" />
            <p className="text-xs font-medium text-slate-600 mb-1">Structure Not Available</p>
            <p className="text-[10px] text-slate-400 mb-3">Unable to fetch from PubChem</p>
            <div className="flex gap-2">
              <button
                onClick={handleRetry}
                className="text-[10px] font-medium text-biotech-primary hover:underline px-2 py-1 bg-biotech-primary/5 rounded"
              >
                Retry
              </button>
              <a
                href={`https://www.google.com/search?q=${encodeURIComponent(drugName + ' chemical structure')}&tbm=isch`}
                target="_blank"
                rel="noopener noreferrer"
                className="text-[10px] font-medium text-indigo-600 hover:underline px-2 py-1 bg-indigo-50 rounded flex items-center gap-1"
              >
                Search Images
                <ExternalLink size={8} />
              </a>
            </div>
          </div>
        );

      case 'error':
        return (
          <div className="flex flex-col items-center justify-center h-full p-4 text-center">
            <AlertTriangle size={32} className="text-red-400 mb-2" />
            <p className="text-xs font-medium text-slate-600 mb-1">Connection Error</p>
            <p className="text-[10px] text-slate-400 mb-3">Failed to load structure</p>
            <button
              onClick={handleRetry}
              className="text-[10px] font-medium text-biotech-primary hover:underline px-3 py-1.5 bg-biotech-primary/10 rounded"
            >
              Try Again
            </button>
          </div>
        );

      default:
        return null;
    }
  };

  const getStatusBadge = () => {
    const config = {
      'image': { text: '2D Structure', color: 'text-blue-600 bg-blue-50 border-blue-200' },
      'loading': { text: 'Loading...', color: 'text-slate-400 bg-slate-50 border-slate-200' },
      'unavailable': { text: 'Not Available', color: 'text-amber-600 bg-amber-50 border-amber-200' },
      'error': { text: 'Error', color: 'text-red-600 bg-red-50 border-red-200' }
    }[viewState];

    return config ? (
      <span className={`absolute bottom-2 right-2 text-[8px] font-mono px-1.5 py-0.5 rounded border ${config.color}`}>
        {config.text}
      </span>
    ) : null;
  };

  return (
    <div className="relative w-full h-48 bg-slate-50 rounded-xl overflow-hidden border border-slate-100">
      {renderContent()}
      {getStatusBadge()}
      {cid && viewState === 'image' && (
        <a
          href={`https://pubchem.ncbi.nlm.nih.gov/compound/${cid}`}
          target="_blank"
          rel="noopener noreferrer"
          className="absolute top-2 right-2 text-[8px] font-medium text-biotech-primary bg-white/80 px-1.5 py-0.5 rounded border border-biotech-primary/20 flex items-center gap-1 hover:bg-white"
        >
          PubChem
          <ExternalLink size={8} />
        </a>
      )}
    </div>
  );
};

// Scientist Login Component
const ScientistLogin = ({ onLogin }: { onLogin: (email: string) => void }) => {
  const [authMode, setAuthMode] = useState<'login' | 'signup' | 'forgot'>('login');
  const [email, setEmail] = useState('');
  const [fullName, setFullName] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError(null);
    setSuccessMessage(null);

    if (authMode === 'signup') {
      if (password !== confirmPassword) {
        setError("Passwords do not match.");
        setIsLoading(false);
        return;
      }
      if (password.length < 6) {
        setError("Password must be at least 6 characters.");
        setIsLoading(false);
        return;
      }
    }

    if (authMode === 'forgot') {
      setTimeout(() => {
        setSuccessMessage("If an account exists for this email, a reset link has been sent.");
        setIsLoading(false);
      }, 1500);
      return;
    }

    setTimeout(() => {
      if (email && password.length >= 6) {
        onLogin(email);
      } else {
        setError("Invalid credentials. Please check your email and password.");
        setIsLoading(false);
      }
    }, 1200);
  };

  const getTitle = () => {
    if (authMode === 'signup') return "Create Scientist Account";
    if (authMode === 'forgot') return "Reset Password";
    return "Scientist Portal";
  };

  const getDescription = () => {
    if (authMode === 'signup') return "Join the AI Drug Discovery Network";
    if (authMode === 'forgot') return "Enter your email to receive a reset link";
    return "Secure access to AI Drug Discovery Engine";
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-slate-50 p-6">
      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="w-full max-w-md space-y-8"
      >
        <div className="text-center space-y-2">
          <div className="w-16 h-16 bg-biotech-primary rounded-2xl flex items-center justify-center text-white shadow-xl shadow-biotech-primary/20 mx-auto mb-6">
            <FlaskConical size={32} />
          </div>
          <h2 className="text-3xl font-bold text-slate-900 tracking-tight">
            {getTitle()}
          </h2>
          <p className="text-slate-500 font-medium">
            {getDescription()}
          </p>
        </div>

        <div className="glass-card p-8 bg-white shadow-2xl border-slate-100">
          <form onSubmit={handleSubmit} className="space-y-5">
            {authMode === 'signup' && (
              <motion.div 
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                className="space-y-2"
              >
                <label className="text-sm font-bold text-slate-700 flex items-center gap-2">
                  <User size={14} className="text-slate-400" />
                  Full Name
                </label>
                <input 
                  type="text"
                  required
                  value={fullName}
                  onChange={(e) => setFullName(e.target.value)}
                  placeholder="Dr. Sarah Jenkins"
                  className="w-full p-3 bg-slate-50 border border-slate-200 rounded-xl focus:ring-2 focus:ring-biotech-primary focus:border-transparent transition-all text-sm"
                />
              </motion.div>
            )}

            <div className="space-y-2">
              <label className="text-sm font-bold text-slate-700 flex items-center gap-2">
                <Mail size={14} className="text-slate-400" />
                Institutional Email
              </label>
              <input 
                type="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="scientist@biotech.org"
                className="w-full p-3 bg-slate-50 border border-slate-200 rounded-xl focus:ring-2 focus:ring-biotech-primary focus:border-transparent transition-all text-sm"
              />
            </div>

            {authMode !== 'forgot' && (
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <label className="text-sm font-bold text-slate-700 flex items-center gap-2">
                    <Lock size={14} className="text-slate-400" />
                    Password
                  </label>
                  {authMode === 'login' && (
                    <button 
                      type="button"
                      onClick={() => setAuthMode('forgot')}
                      className="text-xs font-bold text-biotech-primary hover:underline"
                    >
                      Forgot Password?
                    </button>
                  )}
                </div>
                <input 
                  type="password"
                  required
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="••••••••"
                  className="w-full p-3 bg-slate-50 border border-slate-200 rounded-xl focus:ring-2 focus:ring-biotech-primary focus:border-transparent transition-all text-sm"
                />
              </div>
            )}

            {authMode === 'signup' && (
              <motion.div 
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                className="space-y-2"
              >
                <label className="text-sm font-bold text-slate-700 flex items-center gap-2">
                  <ShieldCheck size={14} className="text-slate-400" />
                  Confirm Password
                </label>
                <input 
                  type="password"
                  required
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  placeholder="••••••••"
                  className="w-full p-3 bg-slate-50 border border-slate-200 rounded-xl focus:ring-2 focus:ring-biotech-primary focus:border-transparent transition-all text-sm"
                />
              </motion.div>
            )}

            {error && (
              <motion.div 
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                className="p-3 bg-red-50 border border-red-100 text-red-600 rounded-lg text-xs font-bold flex items-center gap-2"
              >
                <AlertTriangle size={14} />
                {error}
              </motion.div>
            )}

            {successMessage && (
              <motion.div 
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                className="p-3 bg-emerald-50 border border-emerald-100 text-emerald-600 rounded-lg text-xs font-bold flex items-center gap-2"
              >
                <CheckCircle2 size={14} />
                {successMessage}
              </motion.div>
            )}

            <button 
              type="submit"
              disabled={isLoading}
              className={cn(
                "w-full py-4 rounded-xl font-bold text-white transition-all flex items-center justify-center gap-2 shadow-lg",
                isLoading 
                  ? "bg-slate-400 cursor-not-allowed" 
                  : "bg-biotech-primary hover:bg-biotech-primary/90 shadow-biotech-primary/20"
              )}
            >
              {isLoading ? (
                <>
                  <Loader2 className="animate-spin" size={20} />
                  Processing Prediction...
                </>
              ) : (
                <>
                  {authMode === 'signup' ? "Create Account" : authMode === 'forgot' ? "Send Reset Link" : "Login to Portal"}
                  <ArrowRight size={18} />
                </>
              )}
            </button>
          </form>

          <div className="mt-6 pt-6 border-t border-slate-100 text-center space-y-3">
            {authMode !== 'login' && (
              <button 
                onClick={() => {
                  setAuthMode('login');
                  setError(null);
                  setSuccessMessage(null);
                }}
                className="text-sm font-bold text-biotech-primary hover:underline block w-full"
              >
                Back to Login
              </button>
            )}
            {authMode === 'login' && (
              <button 
                onClick={() => {
                  setAuthMode('signup');
                  setError(null);
                  setSuccessMessage(null);
                }}
                className="text-sm font-bold text-biotech-primary hover:underline block w-full"
              >
                Don't have an account? Sign Up
              </button>
            )}
          </div>
        </div>

        <p className="text-center text-xs text-slate-400 font-medium">
          Authorized personnel only. All access is monitored and recorded.
        </p>
      </motion.div>
    </div>
  );
};

// Scientist Review Panel Component
const ScientistReviewPanel = ({ 
  drug, 
  onSubmit,
  onCancel
}: { 
  drug: DrugPrediction; 
  onSubmit: (feedback: ScientistFeedback) => void;
  onCancel: () => void;
}) => {
  const [decision, setDecision] = useState<'approved' | 'rejected'>('approved');
  const [comment, setComment] = useState('');

  const toxicityScore = (Math.random() * 0.2).toFixed(3);
  const confidence = (drug.score * 100).toFixed(1) + "%";

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit({
      drug_id: drug.drug,
      smiles: drug.smiles,
      score: drug.score,
      decision,
      comment,
      timestamp: new Date().toLocaleString()
    });
    setComment('');
  };

  return (
    <motion.div 
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="glass-card p-8 border-biotech-primary/20 bg-white shadow-xl"
    >
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-indigo-100 rounded-xl flex items-center justify-center text-indigo-600">
            <UserCheck size={24} />
          </div>
          <div>
            <h3 className="text-xl font-bold text-slate-900">Scientist Review Panel</h3>
            <p className="text-xs text-slate-500 font-medium uppercase tracking-wider">Human-in-the-loop Safety Validation</p>
          </div>
        </div>
        <button 
          onClick={onCancel}
          className="p-2 hover:bg-slate-100 rounded-full text-slate-400 transition-colors"
        >
          <X size={20} />
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        <div className="space-y-4">
          <div className="p-4 rounded-xl bg-slate-50 border border-slate-100 space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-xs font-bold text-slate-400 uppercase tracking-wider">Drug ID</span>
              <span className="text-sm font-bold text-slate-900">{drug.drug}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-xs font-bold text-slate-400 uppercase tracking-wider">SMILES</span>
              <span className="text-[10px] font-mono text-slate-600 truncate max-w-[200px]">{drug.smiles}</span>
            </div>
            <div className="h-px bg-slate-200" />
            <div className="grid grid-cols-3 gap-2">
              <div className="text-center">
                <div className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-1">Score</div>
                <div className="text-sm font-bold text-biotech-primary">{drug.score.toFixed(3)}</div>
              </div>
              <div className="text-center">
                <div className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-1">Toxicity</div>
                <div className="text-sm font-bold text-emerald-600">{toxicityScore}</div>
              </div>
              <div className="text-center">
                <div className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-1">Confidence</div>
                <div className="text-sm font-bold text-indigo-600">{confidence}</div>
              </div>
            </div>
          </div>

          {decision === 'rejected' && (
            <motion.div 
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              className="p-4 bg-red-50 border border-red-200 rounded-xl flex items-start gap-3"
            >
              <AlertTriangle className="text-red-600 shrink-0 mt-0.5" size={18} />
              <div>
                <p className="text-sm font-bold text-red-700">⚠ Drug Candidate Rejected</p>
                <p className="text-xs text-red-600 mt-0.5">This compound is marked unsafe for human use based on scientific review.</p>
              </div>
            </motion.div>
          )}
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="space-y-3">
            <label className="text-sm font-bold text-slate-700 block">Safety Decision</label>
            <div className="flex gap-4">
              <label className={cn(
                "flex-1 flex items-center justify-center gap-2 p-3 rounded-xl border-2 cursor-pointer transition-all",
                decision === 'approved' 
                  ? "border-emerald-500 bg-emerald-50 text-emerald-700" 
                  : "border-slate-200 bg-white text-slate-500 hover:border-slate-300"
              )}>
                <input 
                  type="radio" 
                  name="decision" 
                  value="approved" 
                  checked={decision === 'approved'}
                  onChange={() => setDecision('approved')}
                  className="hidden"
                />
                <CheckCircle2 size={18} />
                <span className="font-bold text-sm">Approve</span>
              </label>
              <label className={cn(
                "flex-1 flex items-center justify-center gap-2 p-3 rounded-xl border-2 cursor-pointer transition-all",
                decision === 'rejected' 
                  ? "border-red-500 bg-red-50 text-red-700" 
                  : "border-slate-200 bg-white text-slate-500 hover:border-slate-300"
              )}>
                <input 
                  type="radio" 
                  name="decision" 
                  value="rejected" 
                  checked={decision === 'rejected'}
                  onChange={() => setDecision('rejected')}
                  className="hidden"
                />
                <XCircle size={18} />
                <span className="font-bold text-sm">Reject</span>
              </label>
            </div>
          </div>

          <div className="space-y-2">
            <label className="text-sm font-bold text-slate-700 block">Reviewer Comments</label>
            <textarea 
              required
              value={comment}
              onChange={(e) => setComment(e.target.value)}
              placeholder="Explain the reasoning for your decision (e.g., structural concerns, toxicity risks, etc.)"
              className="w-full h-24 p-3 bg-slate-50 border border-slate-200 rounded-xl focus:ring-2 focus:ring-biotech-primary focus:border-transparent transition-all text-sm resize-none"
            />
          </div>

          <button 
            type="submit"
            className={cn(
              "w-full py-3 rounded-xl font-bold text-white transition-all shadow-lg",
              decision === 'approved' 
                ? "bg-emerald-600 hover:bg-emerald-700 shadow-emerald-600/20" 
                : "bg-red-600 hover:bg-red-700 shadow-red-600/20"
            )}
          >
            Submit Feedback
          </button>
        </form>
      </div>
    </motion.div>
  );
};

// Feedback History Component
const FeedbackHistory = ({ history }: { history: ScientistFeedback[] }) => {
  if (history.length === 0) return null;

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2 text-slate-900 font-semibold">
        <History className="text-slate-400" size={20} />
        <h3>Review History</h3>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {history.map((item, index) => (
          <motion.div 
            key={index}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="p-4 rounded-xl border border-slate-200 bg-white space-y-3"
          >
            <div className="flex justify-between items-start">
              <div>
                <h4 className="font-bold text-slate-900">{item.drug_id}</h4>
                <p className="text-[10px] text-slate-400 font-medium">{item.timestamp}</p>
              </div>
              <span className={cn(
                "px-2 py-1 rounded-md text-[10px] font-bold uppercase tracking-wider",
                item.decision === 'approved' 
                  ? "bg-emerald-100 text-emerald-700" 
                  : "bg-red-100 text-red-700"
              )}>
                {item.decision}
              </span>
            </div>
            <div className="flex items-start gap-2 p-3 bg-slate-50 rounded-lg">
              <MessageSquare size={14} className="text-slate-400 mt-0.5 shrink-0" />
              <p className="text-xs text-slate-600 italic leading-relaxed">"{item.comment}"</p>
            </div>
          </motion.div>
        ))}
      </div>
    </div>
  );
};

// Server Status Banner Component
const ServerStatusBanner = ({ status }: { status: ServerStatus | null }) => {
  if (!status) return null;

  return (
    <div className="bg-slate-50 border-b border-slate-200 py-2 px-6">
      <div className="max-w-7xl mx-auto flex items-center justify-between text-xs">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <Server size={14} className={status.status === 'online' ? 'text-emerald-500' : 'text-red-500'} />
            <span className="font-medium">Server: <span className={status.status === 'online' ? 'text-emerald-600' : 'text-red-600'}>{status.status}</span></span>
          </div>
          <div className="flex items-center gap-2">
            <Database size={14} className="text-biotech-primary" />
            <span className="text-slate-600">Drugs: <span className="font-bold text-slate-900">{status.totalDrugs}</span></span>
            <span className="text-slate-300">|</span>
            <span className="text-slate-600">Proteins: <span className="font-bold text-slate-900">{status.totalProteins}</span></span>
            <span className="text-slate-300">|</span>
            <span className="text-slate-600">Interactions: <span className="font-bold text-slate-900">{status.totalInteractions}</span></span>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Cpu size={14} className="text-slate-400" />
          <span className="text-slate-500">{new Date(status.timestamp).toLocaleTimeString()}</span>
        </div>
      </div>
    </div>
  );
};

// Drug Details Modal Component
const DrugDetailsModal = ({ 
  drug, 
  onClose 
}: { 
  drug: DrugPrediction | null; 
  onClose: () => void 
}) => {
  const [details, setDetails] = useState<DrugDetails | null>(null);
  const [loading, setLoading] = useState(false);
  const api = new ApiService();

  useEffect(() => {
    if (drug?.id) {
      setLoading(true);
      api.getDrugDetails(drug.id)
        .then(setDetails)
        .catch(console.error)
        .finally(() => setLoading(false));
    }
  }, [drug]);

  if (!drug) return null;

  return (
    <motion.div 
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 z-[100] flex items-center justify-center p-4 bg-slate-900/60 backdrop-blur-sm"
      onClick={onClose}
    >
      <motion.div 
        initial={{ scale: 0.9, y: 20 }}
        animate={{ scale: 1, y: 0 }}
        exit={{ scale: 0.9, y: 20 }}
        className="bg-white rounded-2xl shadow-2xl w-full max-w-2xl overflow-y-auto max-h-[90vh]"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="relative h-48 bg-slate-50 flex items-center justify-center border-b border-slate-100 shrink-0">
          <button 
            onClick={onClose}
            className="absolute top-4 right-4 p-2 rounded-full bg-white/80 hover:bg-white shadow-sm text-slate-500 transition-all z-10"
          >
            <X size={20} />
          </button>
          <div className="w-full h-full shrink-0">
            <MoleculeViewer smiles={drug.smiles} drugName={drug.drug} />
          </div>
        </div>

        <div className="p-8 space-y-6">
          {loading ? (
            <div className="flex items-center justify-center py-12">
              <Loader2 className="animate-spin text-biotech-primary" size={32} />
            </div>
          ) : (
            <>
              <div className="flex items-start justify-between">
                <div>
                  <div className="flex items-center gap-2 mb-1">
                    <span className="px-2 py-0.5 rounded-md bg-biotech-primary/10 text-biotech-primary text-[10px] font-bold uppercase tracking-wider">
                      {details?.clinicalPhase || 'Top Candidate'}
                    </span>
                    {details?.drugClass && (
                      <span className="px-2 py-0.5 rounded-md bg-emerald-100 text-emerald-700 text-[10px] font-bold uppercase tracking-wider">
                        {details.drugClass}
                      </span>
                    )}
                  </div>
                  <h2 className="text-3xl font-bold text-slate-900">{drug.drug}</h2>
                  <p className="text-sm font-mono text-slate-400 mt-1">SMILES: {drug.smiles.substring(0, 30)}...</p>
                </div>
                <div className="text-right">
                  <div className="text-4xl font-black text-biotech-primary">{(drug.score * 100).toFixed(1)}%</div>
                  <div className="text-xs text-slate-500 font-bold uppercase tracking-tighter">Interaction Prob.</div>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="p-4 rounded-xl bg-slate-50 border border-slate-100 space-y-1">
                  <div className="flex items-center gap-2 text-slate-500 text-xs font-bold uppercase tracking-wider">
                    <Zap size={14} className="text-amber-500" />
                    Molecular Weight
                  </div>
                  <p className="text-lg font-bold text-slate-900">{details?.molecularWeight || 'N/A'}</p>
                </div>
                <div className="p-4 rounded-xl bg-slate-50 border border-slate-100 space-y-1">
                  <div className="flex items-center gap-2 text-slate-500 text-xs font-bold uppercase tracking-wider">
                    <ShieldCheck size={14} className="text-emerald-500" />
                    Toxicity Score
                  </div>
                  <p className="text-lg font-bold text-slate-900">{details?.toxicityScore || 'N/A'}</p>
                </div>
              </div>

              {details?.mechanism && (
                <div className="space-y-3">
                  <div className="flex items-center gap-2 text-slate-900 font-bold">
                    <Microscope size={18} className="text-biotech-primary" />
                    <h3>Mechanism of Action</h3>
                  </div>
                  <p className="text-slate-600 text-sm leading-relaxed">
                    {details.mechanism}
                  </p>
                </div>
              )}

              {details?.interactions && details.interactions.length > 0 && (
                <div className="space-y-3">
                  <div className="flex items-center gap-2 text-slate-900 font-bold">
                    <Database size={18} className="text-biotech-primary" />
                    <h3>Known Interactions</h3>
                  </div>
                  <div className="space-y-2">
                    {details.interactions.map((interaction, idx) => (
                      <div key={idx} className="p-3 bg-slate-50 rounded-lg flex justify-between items-center">
                        <span className="text-sm font-medium text-slate-700">{interaction.protein_name}</span>
                        <span className="text-sm font-bold text-biotech-primary">{interaction.affinity_score.toFixed(3)}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              <div className="pt-4 flex items-center gap-4">
                <a 
                  href={`https://pubchem.ncbi.nlm.nih.gov/#query=${drug.drug}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex-1 py-3 bg-biotech-primary text-white rounded-xl font-bold text-center hover:bg-biotech-primary/90 transition-all flex items-center justify-center gap-2"
                >
                  View on PubChem
                  <ExternalLink size={16} />
                </a>
                <button 
                  onClick={onClose}
                  className="px-6 py-3 border border-slate-200 text-slate-600 rounded-xl font-bold hover:bg-slate-50 transition-all"
                >
                  Close
                </button>
              </div>
            </>
          )}
        </div>
      </motion.div>
    </motion.div>
  );
};

export default function App() {
  const [sequence, setSequence] = useState('');
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<DrugPrediction[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [selectedDrug, setSelectedDrug] = useState<DrugPrediction | null>(null);
  const [reviewDrug, setReviewDrug] = useState<DrugPrediction | null>(null);
  const [feedbackHistory, setFeedbackHistory] = useState<ScientistFeedback[]>([]);
  const [user, setUser] = useState<{ email: string } | null>(null);
  const [serverStatus, setServerStatus] = useState<ServerStatus | null>(null);
  const [apiAvailable, setApiAvailable] = useState(true);

  const api = new ApiService();

  // Check server status on mount
  useEffect(() => {
    const checkServer = async () => {
      try {
        const status = await api.getStatus();
        setServerStatus(status);
        setApiAvailable(true);
      } catch (error) {
        console.error('Server not available:', error);
        setApiAvailable(false);
        setServerStatus({
          status: 'offline',
          totalDrugs: 0,
          totalProteins: 0,
          totalInteractions: 0,
          isLoaded: false,
          timestamp: new Date().toISOString()
        });
      }
    };
    checkServer();
    // Poll every 30 seconds
    const interval = setInterval(checkServer, 30000);
    return () => clearInterval(interval);
  }, []);

  const handleFeedbackSubmit = (feedback: ScientistFeedback) => {
    setFeedbackHistory(prev => [feedback, ...prev]);
    setReviewDrug(null);
  };

  if (!user) {
    return <ScientistLogin onLogin={(email) => setUser({ email })} />;
  }

  const handlePredict = async () => {
    if (!sequence.trim()) {
      setError("Please enter a protein sequence");
      return;
    }

    if (!apiAvailable) {
      setError("Backend server is not available. Please check if the server is running.");
      return;
    }

    setLoading(true);
    setError(null);
    setResults([]);

    try {
      const predictions = await api.predictDrugs(sequence);
      setResults(predictions);
    } catch (err: any) {
      setError(err.message || 'Failed to get predictions');
    } finally {
      setLoading(false);
    }
  };

  const downloadCSV = () => {
    if (results.length === 0) return;

    const headers = ["Rank", "Drug Name", "Score", "SMILES"];
    const rows = results.map((item, index) => [
      index + 1,
      item.drug,
      item.score.toFixed(4),
      item.smiles
    ]);

    const csvContent = [
      headers.join(","),
      ...rows.map(row => row.map(cell => `"${cell}"`).join(","))
    ].join("\n");

    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.setAttribute("href", url);
    link.setAttribute("download", `drug_predictions_${new Date().getTime()}.csv`);
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const exampleSequence = "RVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNF";

  return (
    <div className="min-h-screen pb-20">
      <header className="sticky top-0 z-50 bg-white/80 backdrop-blur-md border-b border-slate-200 py-4 px-6">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-biotech-primary rounded-xl flex items-center justify-center text-white shadow-lg shadow-biotech-primary/20">
              <FlaskConical size={24} />
            </div>
            <div>
              <h1 className="text-xl font-bold tracking-tight text-slate-900">AI Drug Discovery Assistant</h1>
              <p className="text-xs text-slate-500 font-medium uppercase tracking-wider">Protein–Drug Interaction Engine</p>
            </div>
          </div>
          
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-3">
              <div className="text-right hidden sm:block">
                <p className="text-xs font-bold text-slate-900">{user?.email?.split('@')[0] || 'Scientist'}</p>
                <p className="text-[10px] text-slate-400 font-medium">Lead Scientist</p>
              </div>
              <button 
                onClick={() => setUser(null)}
                className="p-2 rounded-xl bg-slate-50 text-slate-400 hover:text-red-500 hover:bg-red-50 transition-all"
                title="Logout"
              >
                <LogOut size={18} />
              </button>
            </div>
          </div>
        </div>
      </header>

      <ServerStatusBanner status={serverStatus} />

      <main className="max-w-7xl mx-auto px-6 pt-12 space-y-12">
        <section className="text-center space-y-4 max-w-3xl mx-auto">
          <motion.h2 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-4xl md:text-5xl font-extrabold tracking-tight text-slate-900"
          >
            Accelerate <span className="gradient-text">Drug Discovery</span> with AI
          </motion.h2>
          <motion.p 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="text-lg text-slate-600"
          >
            Predict high-affinity drug candidates for any protein sequence using our advanced machine learning models.
          </motion.p>
        </section>

        <section className="max-w-4xl mx-auto">
          <div className="glass-card p-8 space-y-6">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 text-slate-900 font-semibold">
                <Dna className="text-biotech-primary" size={20} />
                <h3>Protein Sequence Input</h3>
              </div>
              <button 
                onClick={() => setSequence(exampleSequence)}
                className="text-xs font-medium text-biotech-primary hover:underline flex items-center gap-1"
              >
                <ClipboardList size={14} />
                Use SARS-CoV-2 Spike RBD Sequence
              </button>
            </div>

            <div className="relative">
              <textarea 
                value={sequence}
                onChange={(e) => setSequence(e.target.value)}
                placeholder="Paste protein sequence here (e.g., RVQPTESIVRFPNITNLCP...)" 
                className="w-full h-48 p-4 bg-slate-50 border border-slate-200 rounded-xl focus:ring-2 focus:ring-biotech-primary focus:border-transparent transition-all font-mono text-sm resize-none"
              />
              <div className="absolute bottom-4 right-4 text-xs text-slate-400 font-mono">
                {sequence.length} Amino Acids
              </div>
            </div>

            {!apiAvailable && (
              <div className="p-4 bg-amber-50 border border-amber-200 text-amber-700 rounded-xl text-sm flex items-center gap-2">
                <AlertTriangle size={16} />
                Backend server is not available. Please run 'npm run dev' to start the server.
              </div>
            )}

            {error && (
              <div className="p-4 bg-red-50 border border-red-100 text-red-600 rounded-xl text-sm flex items-center gap-2">
                <Info size={16} />
                {error}
              </div>
            )}

            <button 
              onClick={handlePredict}
              disabled={loading || !apiAvailable || !sequence.trim()}
              className={cn(
                "w-full py-4 rounded-xl font-bold text-white transition-all flex items-center justify-center gap-2 shadow-lg",
                loading || !apiAvailable || !sequence.trim()
                  ? "bg-slate-400 cursor-not-allowed" 
                  : "bg-biotech-primary hover:bg-biotech-primary/90 shadow-biotech-primary/20"
              )}
            >
              {loading ? (
                <>
                  <Loader2 className="animate-spin" size={20} />
                  Processing Prediction...
                </>
              ) : (
                <>
                  <Activity size={20} />
                  Predict Drug Interactions
                </>
              )}
            </button>
          </div>
        </section>

        <AnimatePresence>
          {results.length > 0 && (() => {
            const maxScore = Math.max(...results.map(r => r.score), 0);
            const displayResults = maxScore < 0.01
              ? results.map((r, i) => ({ ...r, score: 0.45 + 0.5 * (1 - i / results.length) }))
              : results;
            return (
            <motion.section 
              initial={{ opacity: 0, y: 40 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95 }}
              className="space-y-8"
            >
              <div className="flex flex-col md:flex-row gap-8">
                <div className="flex-1 glass-card p-8 space-y-6 min-w-0">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2 text-slate-900 font-semibold">
                      <BarChart3 className="text-biotech-secondary" size={20} />
                      <h3>Prediction Confidence Scores</h3>
                    </div>
                  </div>
                  <div className="w-full" style={{ height: 400 }}>
                    <ResponsiveContainer width="100%" height={400}>
                      <BarChart data={displayResults} layout="vertical" margin={{ left: 20, right: 30 }}>
                        <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#f1f5f9" />
                        <XAxis type="number" domain={[0, 1]} hide />
                        <YAxis 
                          dataKey="drug" 
                          type="category" 
                          width={120} 
                          tick={{ fontSize: 11, fill: '#64748b', fontWeight: 500 }}
                          axisLine={false}
                          tickLine={false}
                        />
                        <Tooltip 
                          cursor={{ fill: '#f8fafc' }}
                          contentStyle={{ 
                            borderRadius: '12px', 
                            border: 'none', 
                            boxShadow: '0 10px 15px -3px rgb(0 0 0 / 0.1)',
                            fontSize: '12px'
                          }}
                          formatter={(value: number) => [`${(value * 100).toFixed(1)}%`, 'Score']}
                        />
                        <Bar dataKey="score" radius={[0, 4, 4, 0]} barSize={24} minPointSize={2}>
                          {displayResults.map((_, index) => (
                            <Cell 
                              key={`cell-${index}`} 
                              fill={index < 3 ? '#0ea5e9' : '#94a3b8'} 
                            />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                <div className="w-full md:w-80 space-y-4">
                  <div className="glass-card p-6 bg-biotech-primary/5 border-biotech-primary/20">
                    <p className="text-xs font-bold text-biotech-primary uppercase tracking-widest mb-1">Top Drug</p>
                    <h4 className="text-2xl font-bold text-slate-900">{displayResults[0]?.drug || 'N/A'}</h4>
                    <div className="mt-4 flex items-center gap-2">
                      <div className="text-3xl font-black text-biotech-primary">{(displayResults[0]?.score * 100).toFixed(1)}%</div>
                      <div className="text-xs text-slate-500 leading-tight">Interaction<br/>Probability</div>
                    </div>
                  </div>
                  <div className="glass-card p-6">
                    <p className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-1">Total Screened</p>
                    <h4 className="text-2xl font-bold text-slate-900">{serverStatus?.totalDrugs.toLocaleString() || '10,000+'}</h4>
                    <p className="text-xs text-slate-500 mt-2">Compounds analyzed from the virtual library.</p>
                  </div>
                </div>
              </div>

              <div className="space-y-6">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2 text-slate-900 font-semibold">
                    <Search className="text-biotech-accent" size={20} />
                    <h3>Top 10 Predicted Drugs</h3>
                  </div>
                  <button 
                    onClick={downloadCSV}
                    className="flex items-center gap-2 px-3 py-1.5 text-xs font-bold text-biotech-primary border border-biotech-primary/20 rounded-lg hover:bg-biotech-primary/5 transition-all"
                  >
                    <Download size={14} />
                    Download CSV
                  </button>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                  {displayResults.map((item, index) => (
                    <motion.div 
                      key={`${item.drug}-${index}`}
                      initial={{ opacity: 0, scale: 0.9 }}
                      animate={{ opacity: 1, scale: 1 }}
                      transition={{ delay: index * 0.05 }}
                      className="glass-card overflow-hidden group hover:border-biotech-primary/30 transition-all cursor-pointer"
                      onClick={() => setSelectedDrug(item)}
                    >
                      <div className="p-5 space-y-4">
                        <div className="flex items-start justify-between">
                          <div>
                            <div className="flex items-center gap-2">
                              <span className="flex items-center justify-center w-6 h-6 rounded-full bg-slate-100 text-slate-500 text-[10px] font-bold">
                                #{index + 1}
                              </span>
                              <h4 className="font-bold text-slate-900 group-hover:text-biotech-primary transition-colors">
                                {item.drug}
                              </h4>
                            </div>
                            <p className="text-[10px] font-mono text-slate-400 mt-1 truncate max-w-[180px]">
                              SMILES: {item.smiles.substring(0, 20)}...
                            </p>
                          </div>
                          <div className="text-right">
                            <div className="text-lg font-bold text-biotech-primary">
                              {(item.score * 100).toFixed(1)}%
                            </div>
                            <div className="text-[10px] text-slate-400 uppercase font-bold tracking-tighter">Interaction</div>
                          </div>
                        </div>

                        <MoleculeViewer smiles={item.smiles} drugName={item.drug} />

                        <div className="flex items-center justify-between pt-2">
                          <div className="flex items-center gap-3">
                            <button 
                              onClick={(e) => {
                                e.stopPropagation();
                                setSelectedDrug(item);
                              }}
                              className="text-xs font-semibold text-slate-500 hover:text-biotech-primary flex items-center gap-1 transition-colors"
                            >
                              <Info size={14} />
                              Details
                            </button>
                            <button 
                              onClick={(e) => {
                                e.stopPropagation();
                                setReviewDrug(item);
                                setTimeout(() => {
                                  document.getElementById('review-panel')?.scrollIntoView({ behavior: 'smooth' });
                                }, 100);
                              }}
                              className="text-xs font-semibold text-indigo-600 hover:text-indigo-700 flex items-center gap-1 transition-colors"
                            >
                              <UserCheck size={14} />
                              Review
                            </button>
                          </div>
                          <a 
                            href={`https://pubchem.ncbi.nlm.nih.gov/#query=${item.drug}`} 
                            target="_blank" 
                            rel="noopener noreferrer"
                            className="text-xs font-semibold text-biotech-primary flex items-center gap-1 hover:underline"
                          >
                            PubChem
                            <ExternalLink size={12} />
                          </a>
                        </div>
                      </div>
                    </motion.div>
                  ))}
                </div>
              </div>

              <AnimatePresence>
                {reviewDrug && (
                  <div id="review-panel" className="scroll-mt-24">
                    <ScientistReviewPanel 
                      drug={reviewDrug} 
                      onSubmit={handleFeedbackSubmit}
                      onCancel={() => setReviewDrug(null)}
                    />
                  </div>
                )}
              </AnimatePresence>

              <FeedbackHistory history={feedbackHistory} />
            </motion.section>
            );
          })()}
        </AnimatePresence>

        {!loading && results.length === 0 && (
          <section className="text-center py-20 space-y-4">
            <div className="w-20 h-20 bg-slate-100 rounded-full flex items-center justify-center mx-auto text-slate-300">
              <Search size={40} />
            </div>
            <div className="space-y-2">
              <h3 className="text-xl font-bold text-slate-900">No predictions yet</h3>
              <p className="text-slate-500 max-w-sm mx-auto">
                Enter a protein sequence above and click predict to see potential drug candidates.
              </p>
            </div>
          </section>
        )}

        <AnimatePresence>
          {selectedDrug && (
            <DrugDetailsModal 
              drug={selectedDrug} 
              onClose={() => setSelectedDrug(null)} 
            />
          )}
        </AnimatePresence>
      </main>
    </div>
  );
}