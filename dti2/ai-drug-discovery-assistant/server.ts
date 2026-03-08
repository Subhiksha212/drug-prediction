import express from "express";
import { createServer as createViteServer } from "vite";
import path from "path";
import { fileURLToPath } from "url";
import fs from 'fs';
import csv from 'csv-parser';
import cors from 'cors';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const DATA_DIR = process.cwd();

console.log('📁 Current working directory:', DATA_DIR);
console.log('📁 __dirname:', __dirname);

interface Drug {
  id: string;
  name: string;
  smiles: string;
  target?: string;
  indication?: string;
  category?: string;
  molecular_weight?: number;
  logp?: number;
  hbd?: number;
  hba?: number;
}

interface Protein {
  id: string;
  name: string;
  sequence: string;
  organism?: string;
  function?: string;
}

interface DrugProteinAffinity {
  drug_id: string;
  protein_id: string;
  affinity_score: number;
  binding_energy?: number;
  pKi?: number;
  pIC50?: number;
}

class MLService {
  private drugs: Drug[] = [];
  private proteins: Protein[] = [];
  private affinities: DrugProteinAffinity[] = [];
  private drugMetadata: Map<string, any> = new Map();
  private isLoaded: boolean = false;

  constructor() {
    this.loadAllData();
  }

  async loadAllData() {
    try {
      console.log('Loading ML data files...');
      
      // Try multiple possible paths
      const possiblePaths = [
        DATA_DIR,
        path.join(DATA_DIR, 'src'),
        path.join(DATA_DIR, 'data'),
        path.join(DATA_DIR, '..', '..', 'data'),
        __dirname
      ];

      console.log('Searching in paths:', possiblePaths);

      // Load drugs.csv
      await this.loadDrugs(possiblePaths);
      
      // Load proteins.csv
      await this.loadProteins(possiblePaths);
      
      // Load drug_protein_affinity.csv
      await this.loadAffinities(possiblePaths);
      
      // Load drugs_metadata (12).csv for additional metadata
      await this.loadDrugMetadata(possiblePaths);
      
      this.isLoaded = true;
      console.log(`✅ Loaded ${this.drugs.length} drugs, ${this.proteins.length} proteins, and ${this.affinities.length} affinity records`);
    } catch (error) {
      console.error('Error loading ML data:', error);
    }
  }

  private async loadDrugs(paths: string[]): Promise<void> {
    for (const basePath of paths) {
      const drugsPath = path.join(basePath, 'drugs.csv');
      if (fs.existsSync(drugsPath)) {
        this.drugs = await this.parseCSV(drugsPath);
        console.log(`✅ Loaded ${this.drugs.length} drugs from ${drugsPath}`);
        return;
      }
    }
    console.warn('⚠️ drugs.csv not found, using mock data');
    this.drugs = this.getMockDrugs();
  }

  private async loadProteins(paths: string[]): Promise<void> {
    for (const basePath of paths) {
      const proteinsPath = path.join(basePath, 'proteins.csv');
      if (fs.existsSync(proteinsPath)) {
        this.proteins = await this.parseCSV(proteinsPath);
        console.log(`✅ Loaded ${this.proteins.length} proteins from ${proteinsPath}`);
        return;
      }
    }
    console.warn('⚠️ proteins.csv not found, using mock data');
    this.proteins = this.getMockProteins();
  }

  private async loadAffinities(paths: string[]): Promise<void> {
    for (const basePath of paths) {
      const affinitiesPath = path.join(basePath, 'drug_protein_affinity.csv');
      if (fs.existsSync(affinitiesPath)) {
        this.affinities = await this.parseCSV(affinitiesPath);
        console.log(`✅ Loaded ${this.affinities.length} affinity records from ${affinitiesPath}`);
        return;
      }
    }
    console.warn('⚠️ drug_protein_affinity.csv not found, generating mock affinities');
    this.generateMockAffinities();
  }

  private async loadDrugMetadata(paths: string[]): Promise<void> {
    for (const basePath of paths) {
      const metadataPath = path.join(basePath, 'drugs_metadata (12).csv');
      if (fs.existsSync(metadataPath)) {
        const metadata = await this.parseCSV(metadataPath);
        metadata.forEach(item => {
          this.drugMetadata.set(item.id || item.drug_id, item);
        });
        console.log(`✅ Loaded metadata for ${this.drugMetadata.size} drugs from ${metadataPath}`);
        return;
      }
    }
  }

  private parseCSV(filePath: string): Promise<any[]> {
    return new Promise((resolve, reject) => {
      const results: any[] = [];
      fs.createReadStream(filePath)
        .pipe(csv())
        .on('data', (data) => results.push(data))
        .on('end', () => resolve(results))
        .on('error', (error) => reject(error));
    });
  }

  // Mock data generators (used if CSV files are not found)
  private getMockDrugs(): Drug[] {
    return [
      { id: "DB00001", name: "Remdesivir", smiles: "CCC(CC)COC(=O)N1C=NC2=C1N=CN2C", target: "RNA polymerase", indication: "COVID-19" },
      { id: "DB00002", name: "Favipiravir", smiles: "NC(=O)C1=NC=NC(=O)N1", target: "RNA polymerase", indication: "Influenza" },
      { id: "DB00003", name: "Dexamethasone", smiles: "CC12CCC3C(C1CCC2(C(=O)CO)O)CCC4=CC(=O)C=CC34C", target: "Glucocorticoid receptor", indication: "Inflammation" },
      { id: "DB00004", name: "Ritonavir", smiles: "CC(C)C1=NC(=CS1)CN(C)C(=O)NC(C(C)C)C(=O)NC(CC2=CC=CC=C2)CC(C(CC3=CC=CC=C3)NC(=O)OCC4=CN=CS4)O", target: "HIV protease", indication: "HIV" },
      { id: "DB00005", name: "Lopinavir", smiles: "CC(C)C1=CC=C(C=C1)OCC(=O)NC(CC2=CC=CC=C2)CC(C(CC3=CC=CC=C3)NC(=O)C(C(C)C)N4CCCC4=O)O", target: "HIV protease", indication: "HIV" },
    ];
  }

  private getMockProteins(): Protein[] {
    return [
      { 
        id: "P0DTD1", 
        name: "SARS-CoV-2 Spike Protein", 
        sequence: "MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAIHVSGTNGTKRFDNPVLPFNDGVYFASTEKSNIIRGWIFGTTLDSKTQSLLIVNNATNVVIKVCEFQFCNDPFLGVYYHKNNKSWMESEFRVYSSANNCTFEYVSQPFLMDLEGKQGNFKNLREFVFKNIDGYFKIYSKHTPINLVRDLPQGFSALEPLVDLPIGINITRFQTLLALHRSYLTPGDSSSGWTAGAAAYYVGYLQPRTFLLKYNENGTITDAVDCALDPLSETKCTLKSFTVEKGIYQTSNFRVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNFNFNGLTGTGVLTESNKKFLPFQQFGRDIADTTDAVRDPQTLEILDITPCSFGGVSVITPGTNTSNQVAVLYQDVNCTEVPVAIHADQLTPTWRVYSTGSNVFQTRAGCLIGAEHVNNSYECDIPIGAGICASYQTQTNSPRRARSVASQSIIAYTMSLGAENSVAYSNNSIAIPTNFTISVTTEILPVSMTKTSVDCTMYICGDSTECSNLLLQYGSFCTQLNRALTGIAVEQDKNTQEVFAQVKQIYKTPPIKDFGGFNFSQILPDPSKPSKRSFIEDLLFNKVTLADAGFIKQYGDCLGDIAARDLICAQKFNGLTVLPPLLTDEMIAQYTSALLAGTITSGWTFGAGAALQIPFAMQMAYRFNGIGVTQNVLYENQKLIANQFNSAIGKIQDSLSSTASALGKLQDVVNQNAQALNTLVKQLSSNFGAISSVLNDILSRLDKVEAEVQIDRLITGRLQSLQTYVTQQLIRAAEIRASANLAATKMSECVLGQSKRVDFCGKGYHLMSFPQSAPHGVVFLHVTYVPAQEKNFTTAPAICHDGKAHFPREGVFVSNGTHWFVTQRNFYEPQIITTDNTFVSGNCDVVIGIVNNTVYDPLQPELDSFKEELDKYFKNHTSPDVDLGDISGINASVVNIQKEIDRLNEVAKNLNESLIDLQELGKYEQYIKWPWYIWLGFIAGLIAIVMVTIMLCCMTSCCSCLKGCCSCGSCCKFDEDDSEPVLKGVKLHYT", 
        organism: "SARS-CoV-2" 
      },
      { 
        id: "P03366", 
        name: "HIV-1 Protease", 
        sequence: "PQITLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMNLPGRWKPKMIGGIGGFIKVRQYDQILIEICGHKAIGTVLVGPTPVNIIGRNLLTQIGCTLNF", 
        organism: "HIV-1" 
      },
    ];
  }

  private generateMockAffinities() {
    for (const drug of this.drugs) {
      for (const protein of this.proteins.slice(0, 3)) {
        this.affinities.push({
          drug_id: drug.id,
          protein_id: protein.id,
          affinity_score: Math.random() * 9 + 1,
          binding_energy: -Math.random() * 12 - 5,
        });
      }
    }
  }

  // Public methods for the API
  isDataLoaded(): boolean {
    return this.isLoaded;
  }

  getStats() {
    return {
      totalDrugs: this.drugs.length,
      totalProteins: this.proteins.length,
      totalInteractions: this.affinities.length,
      isLoaded: this.isLoaded
    };
  }

  searchDrugs(query: string): Drug[] {
    const lowercaseQuery = query.toLowerCase();
    return this.drugs.filter(drug => 
      drug.name?.toLowerCase().includes(lowercaseQuery) ||
      drug.target?.toLowerCase().includes(lowercaseQuery) ||
      drug.indication?.toLowerCase().includes(lowercaseQuery)
    ).slice(0, 20);
  }

  getDrugById(id: string): Drug | null {
    const drug = this.drugs.find(d => d.id === id || d.name === id);
    if (drug) {
      const metadata = this.drugMetadata.get(id);
      if (metadata) {
        return { ...drug, ...metadata };
      }
    }
    return drug || null;
  }

  getProteinById(id: string): Protein | null {
    return this.proteins.find(p => p.id === id || p.name === id) || null;
  }

  getDrugInteractions(drugId: string): any[] {
    const interactions = this.affinities.filter(a => a.drug_id === drugId);
    
    return interactions.map(interaction => {
      const protein = this.proteins.find(p => p.id === interaction.protein_id);
      return {
        ...interaction,
        protein_name: protein?.name || 'Unknown',
        protein_organism: protein?.organism
      };
    });
  }

  async predictBindingAffinity(sequence: string): Promise<any[]> {
    // Find similar proteins in our database
    const similarProteins = this.findSimilarProteins(sequence);
    
    // Get drugs that bind to similar proteins
    const predictions = new Map<string, any>();
    
    for (const protein of similarProteins) {
      const interactions = this.affinities.filter(a => a.protein_id === protein.id);
      
      for (const interaction of interactions) {
        const drug = this.drugs.find(d => d.id === interaction.drug_id);
        if (drug) {
          if (!predictions.has(drug.id)) {
            predictions.set(drug.id, {
              drug: drug.name,
              smiles: drug.smiles,
              target: drug.target,
              scores: []
            });
          }
          
          const similarity = protein.similarity || 0.5;
          predictions.get(drug.id).scores.push(interaction.affinity_score * similarity);
        }
      }
    }

    // If no predictions found, return mock data
    if (predictions.size === 0) {
      return this.getMockPredictions();
    }

    // Convert to array and calculate average scores
    const results = Array.from(predictions.entries()).map(([id, data]) => ({
      id,
      drug: data.drug,
      smiles: data.smiles,
      target: data.target,
      score: data.scores.reduce((a: number, b: number) => a + b, 0) / data.scores.length
    }));

    // Sort by score and return
    return results
      .sort((a, b) => b.score - a.score)
      .slice(0, 10)
      .map(({ drug, score, smiles, target }) => ({
        drug,
        score: parseFloat(score.toFixed(2)),
        smiles,
        target
      }));
  }

  private getMockPredictions(): any[] {
    return [
      { drug: "Remdesivir", score: 0.87, smiles: "CCC(CC)COC(=O)N1C=NC2=C1N=CN2C", target: "RNA polymerase" },
      { drug: "Favipiravir", score: 0.82, smiles: "NC(=O)C1=NC=NC(=O)N1", target: "RNA polymerase" },
      { drug: "Dexamethasone", score: 0.78, smiles: "CC12CCC3C(C1CCC2(C(=O)CO)O)CCC4=CC(=O)C=CC34C", target: "Glucocorticoid receptor" },
      { drug: "Ritonavir", score: 0.75, smiles: "CC(C)C1=NC(=CS1)CN(C)C(=O)NC(C(C)C)C(=O)NC(CC2=CC=CC=C2)CC(C(CC3=CC=CC=C3)NC(=O)OCC4=CN=CS4)O", target: "HIV protease" },
      { drug: "Lopinavir", score: 0.72, smiles: "CC(C)C1=CC=C(C=C1)OCC(=O)NC(CC2=CC=CC=C2)CC(C(CC3=CC=CC=C3)NC(=O)C(C(C)C)N4CCCC4=O)O", target: "HIV protease" },
      { drug: "Hydroxychloroquine", score: 0.68, smiles: "CCN(CCO)CCCC(C)NC1=C2C=CC=CC2=NC=C1", target: "TLR signaling" },
      { drug: "Azithromycin", score: 0.65, smiles: "CCC1C(C(C(N(C)C(CC(C(C(C(C(C(=O)O1)C)OC2CC(C(C(O2)C)O)(C)OC)C)C)O)C)C)O)(C)O", target: "50S ribosomal subunit" },
      { drug: "Ivermectin", score: 0.62, smiles: "CC1C(CCC2(O1)CC3CC(O2)CC=C4C3C(C=C4)OC5CC(C(C(O5)C)OC6CC(C(C(O6)C)O)O)C)O)C", target: "Glutamate-gated chloride channel" },
      { drug: "Oseltamivir", score: 0.59, smiles: "CC(CC)C(C)C(=O)OC1C=C(CC(C1NC(=O)C)C=C)CO", target: "Neuraminidase" },
      { drug: "Ribavirin", score: 0.55, smiles: "C1=NC(=NN1C2C(C(C(O2)CO)O)O)C(=O)N", target: "Inosine monophosphate dehydrogenase" },
    ];
  }

  private findSimilarProteins(sequence: string): any[] {
    if (!sequence || sequence.length === 0 || this.proteins.length === 0) {
      return [];
    }

    const k = 3;
    const queryKmers = new Set<string>();

    for (let i = 0; i <= sequence.length - k; i++) {
      queryKmers.add(sequence.substring(i, i + k));
    }

    return this.proteins
      .filter(p => p.sequence && p.sequence.length >= k)
      .map(protein => {
        const proteinKmers = new Set<string>();
        for (let i = 0; i <= protein.sequence.length - k; i++) {
          proteinKmers.add(protein.sequence.substring(i, i + k));
        }

        const intersection = new Set([...queryKmers].filter(x => proteinKmers.has(x)));
        const union = new Set([...queryKmers, ...proteinKmers]);
        const similarity = union.size === 0 ? 0 : intersection.size / union.size;

        return {
          ...protein,
          similarity
        };
      })
      .filter(p => p.similarity > 0.1)
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, 5);
  }

  getSimilarDrugs(drugId: string, limit: number = 5): Drug[] {
    const drug = this.getDrugById(drugId);
    if (!drug) return [];

    return this.drugs
      .filter(d => d.id !== drugId && 
        (d.target === drug.target || d.indication === drug.indication))
      .slice(0, limit);
  }

  getAllDrugs(limit?: number): Drug[] {
    return limit ? this.drugs.slice(0, limit) : this.drugs;
  }

  getAllProteins(limit?: number): Protein[] {
    return limit ? this.proteins.slice(0, limit) : this.proteins;
  }
}

async function startServer() {
  const app = express();
  const PORT = 3000;

  // Add test endpoint first
  app.get("/api/test", (req, res) => {
    const files = fs.readdirSync(DATA_DIR).filter(f => f.endsWith('.csv'));
    res.json({ 
      message: "Server is running", 
      cwd: DATA_DIR,
      files: files,
      serverTime: new Date().toISOString()
    });
  });

  // Initialize ML service
  const mlService = new MLService();

  app.use(express.json());
  app.use(cors());

  // API Routes with error handling
  app.get("/api/status", (req, res) => {
    try {
      res.json({
        status: "online",
        ...mlService.getStats(),
        timestamp: new Date().toISOString()
      });
    } catch (error) {
      res.status(500).json({ error: 'Failed to get status', details: error.message });
    }
  });

  app.get("/api/drugs", (req, res) => {
    try {
      const { limit, search } = req.query;
      
      if (search) {
        const results = mlService.searchDrugs(search as string);
        return res.json(results);
      }
      
      const drugs = mlService.getAllDrugs(limit ? parseInt(limit as string) : undefined);
      res.json(drugs);
    } catch (error) {
      res.status(500).json({ error: 'Failed to fetch drugs', details: error.message });
    }
  });

  app.get("/api/drugs/:id", (req, res) => {
    try {
      const drug = mlService.getDrugById(req.params.id);
      if (!drug) {
        return res.status(404).json({ error: "Drug not found" });
      }
      
      const interactions = mlService.getDrugInteractions(drug.id);
      
      res.json({
        ...drug,
        interactions
      });
    } catch (error) {
      res.status(500).json({ error: 'Failed to fetch drug details', details: error.message });
    }
  });

  app.get("/api/drugs/:id/similar", (req, res) => {
    try {
      const { limit } = req.query;
      const similar = mlService.getSimilarDrugs(
        req.params.id,
        limit ? parseInt(limit as string) : 5
      );
      res.json(similar);
    } catch (error) {
      res.status(500).json({ error: 'Failed to fetch similar drugs', details: error.message });
    }
  });

  app.get("/api/proteins", (req, res) => {
    try {
      const { limit } = req.query;
      const proteins = mlService.getAllProteins(limit ? parseInt(limit as string) : undefined);
      res.json(proteins);
    } catch (error) {
      res.status(500).json({ error: 'Failed to fetch proteins', details: error.message });
    }
  });

  app.get("/api/proteins/:id", (req, res) => {
    try {
      const protein = mlService.getProteinById(req.params.id);
      if (!protein) {
        return res.status(404).json({ error: "Protein not found" });
      }
      res.json(protein);
    } catch (error) {
      res.status(500).json({ error: 'Failed to fetch protein', details: error.message });
    }
  });

  app.post("/api/predict", async (req, res) => {
    try {
      const { sequence } = req.body;

      if (!sequence) {
        return res.status(400).json({ 
          success: false,
          error: "Protein sequence is required" 
        });
      }

      if (sequence.length < 5) {
        return res.status(400).json({ 
          success: false,
          error: "Protein sequence must be at least 5 characters" 
        });
      }

      if (sequence.length > 5000) {
        return res.status(400).json({ 
          success: false,
          error: "Protein sequence too long (max 5000 characters)" 
        });
      }

      console.log(`🔬 Processing prediction for sequence length: ${sequence.length}`);
      const predictions = await mlService.predictBindingAffinity(sequence);
      
      res.json(predictions);

    } catch (error) {
      console.error("❌ Prediction API error:", error);
      res.status(500).json({ 
        success: false,
        error: "Prediction failed", 
        details: error instanceof Error ? error.message : String(error)
      });
    }
  });

  app.get("/api/interactions/:drugId", (req, res) => {
    try {
      const interactions = mlService.getDrugInteractions(req.params.drugId);
      res.json(interactions);
    } catch (error) {
      res.status(500).json({ error: 'Failed to fetch interactions', details: error.message });
    }
  });

  // Vite middleware for development
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    app.use(express.static(path.join(__dirname, "dist")));
    app.get("*", (req, res) => {
      res.sendFile(path.join(__dirname, "dist", "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`\n🚀 Server running on http://localhost:${PORT}`);
    console.log(`📊 Test endpoint: http://localhost:${PORT}/api/test`);
    console.log(`📊 API endpoints:`);
    console.log(`   GET  /api/status - Server status`);
    console.log(`   GET  /api/drugs - List all drugs`);
    console.log(`   GET  /api/drugs/:id - Get drug details`);
    console.log(`   GET  /api/drugs/:id/similar - Find similar drugs`);
    console.log(`   GET  /api/proteins - List all proteins`);
    console.log(`   GET  /api/proteins/:id - Get protein details`);
    console.log(`   POST /api/predict - Predict drug-protein binding`);
    console.log(`   GET  /api/interactions/:drugId - Get drug interactions\n`);
  });
}

startServer().catch(error => {
  console.error('Failed to start server:', error);
  process.exit(1);
});