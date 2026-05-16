import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';

export interface AuditReportKpis {
  roas?: number;
  conversions?: number;
  spend?: number;
  [key: string]: unknown;
}

export interface AuditReportPredictedKpis {
  roas_h14?: number;
  conversions_h14?: number;
  cpa_h14?: number;
  ctr_h14?: number;
  cpc_h14?: number;
  [key: string]: unknown;
}

export interface AuditReportTrend {
  roas_trend_pct?: number;
  conversions_trend_pct?: number;
  spend_trend_pct?: number;
}

export interface AuditReportAnomaly {
  date?: string;
  score?: number;
  explanation?: string;
  roas?: number;
  cpa?: number;
  spend?: number;
}

export interface AuditReportCampaign {
  campaign_id?: string;
  global_campaign_id?: string;
  platform?: string;
  status?: string;
  health_score?: number;
  current_kpis?: AuditReportKpis;
  predicted_kpis?: AuditReportPredictedKpis;
  trend?: AuditReportTrend;
  anomalies?: {
    level?: string;
    score?: number;
    n_anomaly_days?: number;
    types?: string[];
    top_anomalies?: AuditReportAnomaly[];
  };
  anomaly_summary?: {
    main_signal?: string;
    main_signal_detail?: string;
    level?: string;
    n_days?: number;
    displayed_days?: number;
    period?: string;
    roas_range?: string;
    max_score?: number;
    types?: string[];
    business_summary?: string;
  };
  root_cause?: {
    code?: string;
    label?: string;
    confidence?: number;
    evidence?: string;
  };
  recommended_action?: {
    action?: string;
    label?: string;
    priority?: string;
    expected_impact?: Record<string, unknown>;
    budget_adjustment?: Record<string, unknown>;
    constraints_applied?: string[];
  };
  business_recommendation?: {
    title?: string;
    summary?: string;
    short_summary?: string;
    advice?: string;
    technical_summary?: string;
  };
  explanations?: {
    xai_summary?: string;
    dashboard_summary?: string;
  };
}

export interface AuditReportData {
  title?: string;
  generated_at?: string;
  executive_summary?: {
    headline?: string;
    key_findings?: string[];
    priority_actions?: Array<{
      campaign_id?: string;
      platform?: string;
      priority?: string;
      action?: string;
      summary?: string;
      short_summary?: string;
      advice?: string;
    }>;
  };
  portfolio_summary?: {
    n_campaigns?: number;
    status_counts?: Record<string, number>;
    platform_counts?: Record<string, number>;
    recommended_action_counts?: Record<string, number>;
    average_health_score?: number;
    total_current_spend?: number;
    cross_channel?: Record<string, unknown>;
  };
  campaigns?: AuditReportCampaign[];
  next_steps?: string[];
}

export interface AuditReportResponse {
  status: string;
  message?: string;
  output_file?: string;
  data?: AuditReportData;
}

@Injectable({ providedIn: 'root' })
export class AuditReportApi {
  private readonly auditReportUrl = 'http://127.0.0.1:8000/audit-report';

  constructor(private readonly http: HttpClient) {}

  generateAuditReport(): Observable<AuditReportResponse> {
    return this.http.post<AuditReportResponse>(this.auditReportUrl, {});
  }
}
