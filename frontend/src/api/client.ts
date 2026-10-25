/**
 * Typed API Client for XAI-RAG Backend.
 * Handles role-based headers, error envelope parsing, and strict typing.
 */

import {
  ApplicantDetail,
  ApplicantSummary,
  AuditJobStatusResponse,
  AuditRecordSummary,
  AuditReport,
  DocumentRecord,
  EvaluationMetricsResponse,
  UserRole,
} from "../types/api";

const BASE_URL = "";

export class ApiError extends Error {
  status: number;
  details?: any;

  constructor(message: string, status: number, details?: any) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.details = details;
  }
}

let activeUserRole: UserRole = "analyst";
const tokenCache: Record<string, string> = {};

const ROLE_CREDENTIALS: Record<string, { username: string; password: string }> = {
  analyst: { username: "analyst", password: "AnalystSecure2026!" },
  reviewer: { username: "reviewer", password: "ReviewerSecure2026!" },
  admin: { username: "admin", password: "AdminSecure2026!" },
};

export async function getAuthTokenForRole(role: string): Promise<string | null> {
  if (tokenCache[role]) return tokenCache[role];
  const creds = ROLE_CREDENTIALS[role] || ROLE_CREDENTIALS.analyst;
  try {
    const res = await fetch(`${BASE_URL}/api/auth/token`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(creds),
    });
    if (res.ok) {
      const data = await res.json();
      tokenCache[role] = data.access_token;
      return data.access_token;
    }
  } catch (err) {
    console.warn("Auth token acquisition failed:", err);
  }
  return null;
}

export function setActiveUserRole(role: UserRole) {
  activeUserRole = role;
}

export function getActiveUserRole(): UserRole {
  return activeUserRole;
}

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const headers = new Headers(options.headers || {});
  if (!path.startsWith("/api/auth/token") && !headers.has("Authorization")) {
    const token = await getAuthTokenForRole(activeUserRole);
    if (token) {
      headers.set("Authorization", `Bearer ${token}`);
    }
  }
  if (!headers.has("X-User-Role")) {
    headers.set("X-User-Role", activeUserRole);
  }

  const response = await fetch(`${BASE_URL}${path}`, {
    ...options,
    headers,
  });

  if (!response.ok) {
    let errorDetail = response.statusText;
    try {
      const errJson = await response.json();
      errorDetail = errJson.message || errJson.detail || JSON.stringify(errJson);
    } catch {
      // ignore
    }
    throw new ApiError(errorDetail, response.status);
  }

  return response.json() as Promise<T>;
}

export const api = {
  async getHealth(): Promise<Record<string, any>> {
    return request<Record<string, any>>("/health");
  },

  async getModels(): Promise<{ active_model: string | null; models: any[] }> {
    return request<{ active_model: string | null; models: any[] }>("/api/models");
  },

  async getApplicants(): Promise<{ total: number; applicants: ApplicantSummary[] }> {
    return request<{ total: number; applicants: ApplicantSummary[] }>("/api/applicants");
  },

  async getApplicant(ref: string): Promise<ApplicantDetail> {
    return request<ApplicantDetail>(`/api/applicants/${encodeURIComponent(ref)}`);
  },

  async runAudit(payload: {
    applicant_data: Record<string, any>;
    applicant_ref?: string;
    idempotency_key?: string;
    run_async?: boolean;
    user_role?: string;
  }): Promise<AuditJobStatusResponse> {
    return request<AuditJobStatusResponse>("/api/audits", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  },

  async getJobStatus(jobId: string): Promise<AuditJobStatusResponse> {
    return request<AuditJobStatusResponse>(`/api/jobs/${encodeURIComponent(jobId)}`);
  },

  async getAuditSummary(auditId: string): Promise<Record<string, any>> {
    return request<Record<string, any>>(`/api/audits/${encodeURIComponent(auditId)}`);
  },

  async getAuditReport(auditId: string): Promise<AuditReport> {
    return request<AuditReport>(`/api/audits/${encodeURIComponent(auditId)}/report`);
  },

  async getAuditTrace(auditId: string): Promise<Record<string, any>> {
    return request<Record<string, any>>(`/api/audits/${encodeURIComponent(auditId)}/trace`);
  },

  getAuditPdfUrl(auditId: string): string {
    return `/api/audits/${encodeURIComponent(auditId)}/export.pdf`;
  },

  async downloadAuditPdf(auditId: string): Promise<Blob> {
    const token = await getAuthTokenForRole(activeUserRole);
    const headers = new Headers({ "X-User-Role": activeUserRole });
    if (token) {
      headers.set("Authorization", `Bearer ${token}`);
    }
    const response = await fetch(`/api/audits/${encodeURIComponent(auditId)}/export.pdf`, {
      headers,
    });
    if (!response.ok) {
      throw new ApiError("Failed to download PDF dossier", response.status);
    }
    return response.blob();
  },

  async recordDisposition(
    auditId: string,
    payload: { disposition: string; notes: string; reviewer_id: string }
  ): Promise<{ status: string; audit_id: string; disposition: string }> {
    return request<{ status: string; audit_id: string; disposition: string }>(
      `/api/audits/${encodeURIComponent(auditId)}/disposition`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      }
    );
  },

  async listAudits(
    limit: number = 50,
    offset: number = 0
  ): Promise<{ total: number; audits: AuditRecordSummary[] }> {
    return request<{ total: number; audits: AuditRecordSummary[] }>(
      `/api/audits?limit=${limit}&offset=${offset}`
    );
  },

  async listDocuments(): Promise<{ total: number; documents: DocumentRecord[] }> {
    return request<{ total: number; documents: DocumentRecord[] }>("/api/documents");
  },

  async uploadDocument(formData: FormData): Promise<any> {
    const token = await getAuthTokenForRole(activeUserRole);
    const headers = new Headers({ "X-User-Role": activeUserRole });
    if (token) {
      headers.set("Authorization", `Bearer ${token}`);
    }
    const response = await fetch("/api/documents", {
      method: "POST",
      headers,
      body: formData,
    });
    if (!response.ok) {
      const errJson = await response.json().catch(() => ({}));
      throw new ApiError(errJson.detail || errJson.message || "Upload failed", response.status);
    }
    return response.json();
  },

  async reindexDocument(docId: string): Promise<any> {
    return request<any>(`/api/documents/${encodeURIComponent(docId)}/reindex`, {
      method: "POST",
    });
  },

  async updateDocumentStatus(docId: string, status: "active" | "superseded"): Promise<any> {
    return request<any>(`/api/documents/${encodeURIComponent(docId)}/status`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ status }),
    });
  },

  async getEvaluationMetrics(): Promise<EvaluationMetricsResponse> {
    return request<EvaluationMetricsResponse>("/api/evaluation/metrics");
  },
};
