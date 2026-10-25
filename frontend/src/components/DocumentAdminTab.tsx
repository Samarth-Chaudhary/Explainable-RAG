import React, { useState, useEffect } from "react";
import { Database, Upload, RefreshCw, FileText, CheckCircle2, AlertTriangle, ShieldCheck } from "lucide-react";
import { DocumentRecord, UserRole } from "../types/api";
import { api } from "../api/client";

interface DocumentAdminTabProps {
  activeRole: UserRole;
}

export const DocumentAdminTab: React.FC<DocumentAdminTabProps> = ({
  activeRole,
}) => {
  const [documents, setDocuments] = useState<DocumentRecord[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [reindexing, setReindexing] = useState<boolean>(false);

  // Upload Form State
  const [file, setFile] = useState<File | null>(null);
  const [sourceId, setSourceId] = useState<string>("");
  const [title, setTitle] = useState<string>("");
  const [issuer, setIssuer] = useState<string>("");
  const [jurisdiction, setJurisdiction] = useState<string>("IN");
  const [sourceFamily, setSourceFamily] = useState<string>("underwriting_rules");
  const [version, setVersion] = useState<string>("1.0");
  const [effectiveDate, setEffectiveDate] = useState<string>("2026-02-01");
  const [uploading, setUploading] = useState<boolean>(false);

  const isAdmin = activeRole === "admin";

  useEffect(() => {
    loadDocuments();
  }, []);

  const loadDocuments = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.listDocuments();
      setDocuments(res.documents);
    } catch (err: any) {
      setError(err.message || "Failed to load document registry.");
    } finally {
      setLoading(false);
    }
  };

  const handleReindex = async (docId: string) => {
    if (!isAdmin) {
      alert("Role 'admin' required to rebuild vector and lexical search indexes.");
      return;
    }
    setReindexing(true);
    try {
      const res = await api.reindexDocument(docId);
      alert(`Index rebuilt successfully! New Index Version: ${res.index_version.slice(0, 16)}...`);
      loadDocuments();
    } catch (err: any) {
      alert(`Reindexing failed: ${err.message}`);
    } finally {
      setReindexing(false);
    }
  };

  const handleToggleStatus = async (doc: DocumentRecord) => {
    if (!isAdmin) {
      alert("Role 'admin' required to modify document active/superseded status.");
      return;
    }
    const newStatus = doc.status === "active" ? "superseded" : "active";
    try {
      await api.updateDocumentStatus(doc.source_id, newStatus);
      loadDocuments();
    } catch (err: any) {
      alert(`Failed to update status: ${err.message}`);
    }
  };

  const handleUploadSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) {
      alert("Please select a valid PDF file.");
      return;
    }
    if (!isAdmin) {
      alert("Document ingestion is restricted to Administrator role.");
      return;
    }

    setUploading(true);
    try {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("source_id", sourceId);
      formData.append("title", title);
      formData.append("issuer", issuer);
      formData.append("jurisdiction", jurisdiction);
      formData.append("source_family", sourceFamily);
      formData.append("version", version);
      formData.append("effective_date", effectiveDate);

      const res = await api.uploadDocument(formData);
      alert(`Document ${res.source_id} successfully parsed and ingested (${res.chunks_extracted} chunks extracted).`);
      setFile(null);
      setSourceId("");
      setTitle("");
      setIssuer("");
      loadDocuments();
    } catch (err: any) {
      alert(`Upload rejected: ${err.message}`);
    } finally {
      setUploading(false);
    }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "20px" }}>
      {/* Upload Card */}
      <div className="card">
        <div className="card-header">
          <div>
            <span className="card-title">
              <Upload size={18} color="#38bdf8" />
              Authoritative Policy Ingestion & Index Pipeline
            </span>
            <span className="card-subtitle">
              Upload PDF circulars, underwriting rules, and regulatory documents with strict header validation
            </span>
          </div>
          {!isAdmin && (
            <span className="badge badge-warning font-mono">
              Read-Only View (Switch to Admin role to upload/reindex)
            </span>
          )}
        </div>

        <div className="card-body">
          <form onSubmit={handleUploadSubmit} style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "14px" }}>
            <div>
              <label style={{ fontSize: "0.75rem", color: "var(--text-secondary)", fontWeight: 600 }}>
                PDF Document File (.pdf)
              </label>
              <input
                type="file"
                accept=".pdf,application/pdf"
                onChange={(e) => setFile(e.target.files ? e.target.files[0] : null)}
                disabled={!isAdmin || uploading}
                required
                style={{ width: "100%", marginTop: "4px", fontSize: "0.8125rem", color: "var(--text-secondary)" }}
              />
            </div>

            <div>
              <label style={{ fontSize: "0.75rem", color: "var(--text-secondary)", fontWeight: 600 }}>
                Unique Source ID
              </label>
              <input
                type="text"
                value={sourceId}
                onChange={(e) => setSourceId(e.target.value)}
                placeholder="e.g. RBI-MD-2026-SEC28"
                disabled={!isAdmin || uploading}
                required
                style={{ width: "100%", marginTop: "4px", padding: "6px 10px", backgroundColor: "var(--bg-surface)", border: "1px solid var(--border-subtle)", borderRadius: "var(--radius-sm)", color: "var(--text-primary)", fontSize: "0.8125rem" }}
              />
            </div>

            <div>
              <label style={{ fontSize: "0.75rem", color: "var(--text-secondary)", fontWeight: 600 }}>
                Document Title
              </label>
              <input
                type="text"
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                placeholder="e.g. Master Direction on LTV & Tenor Limits"
                disabled={!isAdmin || uploading}
                required
                style={{ width: "100%", marginTop: "4px", padding: "6px 10px", backgroundColor: "var(--bg-surface)", border: "1px solid var(--border-subtle)", borderRadius: "var(--radius-sm)", color: "var(--text-primary)", fontSize: "0.8125rem" }}
              />
            </div>

            <div>
              <label style={{ fontSize: "0.75rem", color: "var(--text-secondary)", fontWeight: 600 }}>
                Issuing Authority
              </label>
              <input
                type="text"
                value={issuer}
                onChange={(e) => setIssuer(e.target.value)}
                placeholder="e.g. Reserve Bank of India"
                disabled={!isAdmin || uploading}
                required
                style={{ width: "100%", marginTop: "4px", padding: "6px 10px", backgroundColor: "var(--bg-surface)", border: "1px solid var(--border-subtle)", borderRadius: "var(--radius-sm)", color: "var(--text-primary)", fontSize: "0.8125rem" }}
              />
            </div>

            <div style={{ display: "flex", alignItems: "flex-end", gridColumn: "1 / -1", justifyContent: "flex-end" }}>
              <button
                type="submit"
                className="btn btn-primary"
                disabled={!isAdmin || uploading}
                id="btn-upload-document"
              >
                {uploading ? (
                  <>
                    <RefreshCw size={14} className="animate-spin" />
                    Ingesting & Indexing...
                  </>
                ) : (
                  <>
                    <Upload size={14} />
                    Ingest & Chunk Document
                  </>
                )}
              </button>
            </div>
          </form>
        </div>
      </div>

      {/* Document Registry Table */}
      <div className="card">
        <div className="card-header">
          <div>
            <span className="card-title">
              <Database size={18} color="#34d399" />
              Registered Policy Documents & Ingestion State
            </span>
            <span className="card-subtitle">
              Active policy clauses indexed in hybrid FAISS dense and BM25 lexical stores
            </span>
          </div>
          <button className="btn btn-secondary" onClick={loadDocuments} disabled={loading}>
            <RefreshCw size={14} className={loading ? "animate-spin" : ""} />
            Refresh List
          </button>
        </div>

        <div className="card-body">
          {error && (
            <div style={{ padding: "10px 14px", backgroundColor: "var(--risk-high-bg)", border: "1px solid var(--risk-high-border)", borderRadius: "var(--radius-sm)", color: "#fca5a5", marginBottom: "16px", display: "flex", alignItems: "center", gap: "8px" }}>
              <AlertTriangle size={16} />
              <span>{error}</span>
            </div>
          )}

          {documents.length === 0 && !loading ? (
            <div style={{ padding: "30px", textAlign: "center", color: "var(--text-muted)" }}>
              No policy documents registered. Ingest a policy document to activate hybrid retrieval.
            </div>
          ) : (
            <div className="table-wrap">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Source ID</th>
                    <th>Document Title</th>
                    <th>Issuer</th>
                    <th>Status</th>
                    <th>Pages / Chunks</th>
                    <th>Parser Status</th>
                    <th>Digest (SHA-256)</th>
                    <th>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {documents.map((d) => (
                    <tr key={d.source_id}>
                      <td className="font-mono" style={{ fontWeight: 600, color: "#60a5fa" }}>
                        {d.source_id}
                      </td>
                      <td>
                        <div style={{ fontWeight: 500 }}>{d.title}</div>
                        <div style={{ fontSize: "0.6875rem", color: "var(--text-muted)" }}>
                          Version: {d.version} | Effective: {d.effective_date}
                        </div>
                      </td>
                      <td>{d.issuer}</td>
                      <td>
                        <span
                          className={`badge ${
                            d.status === "active" ? "badge-risk-low" : "badge-neutral"
                          }`}
                        >
                          {d.status.toUpperCase()}
                        </span>
                      </td>
                      <td className="font-mono">
                        {d.total_pages} pages / {d.total_chunks} chunks
                      </td>
                      <td>
                        <span className="badge badge-info font-mono">{d.parser_status}</span>
                      </td>
                      <td className="font-mono text-muted" style={{ fontSize: "0.6875rem" }}>
                        {d.sha256_hash ? d.sha256_hash.slice(0, 12) + "..." : "N/A"}
                      </td>
                      <td>
                        <div style={{ display: "flex", gap: "6px" }}>
                          <button
                            className="btn btn-secondary"
                            style={{ padding: "3px 8px", fontSize: "0.6875rem" }}
                            onClick={() => handleToggleStatus(d)}
                            disabled={!isAdmin}
                          >
                            {d.status === "active" ? "Mark Superseded" : "Activate"}
                          </button>
                          <button
                            className="btn btn-primary"
                            style={{ padding: "3px 8px", fontSize: "0.6875rem" }}
                            onClick={() => handleReindex(d.source_id)}
                            disabled={!isAdmin || reindexing}
                          >
                            Reindex
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
