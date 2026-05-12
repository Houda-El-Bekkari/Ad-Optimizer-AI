import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';

export interface JsonItems<T> {
  items?: T[];
}

export interface CampaignOptimizationRecord {
  id?: number;
  workflow_execution_id?: string;
  global_campaign_id?: string;
  campaign_id: string;
  platform: string;
  health_score?: number;
  health_status?: string;
  anomaly_level?: string;
  anomaly_score?: number;
  n_anomaly_days?: number;
  current_kpis?: unknown;
  predicted_kpis?: unknown;
  trend?: unknown;
  anomaly_types?: unknown;
  top_anomalies?: unknown;
  root_cause?: string;
  root_cause_label?: string;
  causal_confidence?: number;
  recommended_action?: string;
  action_label?: string;
  priority?: string;
  expected_impact?: unknown;
  budget_adjustment?: unknown;
  dashboard_summary?: string;
  source_generated_at?: string;
  created_at?: string;
}

export interface CampaignOptimizationDashboardResponse {
  count: number;
  items: CampaignOptimizationRecord[];
  generated_at: string;
}

@Injectable({
  providedIn: 'root',
})
export class CampaignOptimizationApi {
  private readonly case1WebhookUrl =
    'http://localhost:5678/webhook/case1-optimization-campaign-existing';
  private readonly dashboardWebhookUrl =
    'http://localhost:5678/webhook/case1-optimization-dashboard';

  constructor(private readonly http: HttpClient) {}

  triggerExistingCampaignAudit(): Observable<unknown> {
    return this.http.post(this.case1WebhookUrl, {});
  }

  getOptimizationDashboard(): Observable<CampaignOptimizationDashboardResponse> {
    return this.http.post<CampaignOptimizationDashboardResponse>(this.dashboardWebhookUrl, {});
  }
}
