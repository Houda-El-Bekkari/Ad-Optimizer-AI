import { HttpInterceptorFn } from '@angular/common/http';
import { inject } from '@angular/core';
import { Router } from '@angular/router';
import { catchError, throwError } from 'rxjs';

import { AuthSession } from './services/auth-session';
import { environment } from '../environments/environment';

export const authInterceptor: HttpInterceptorFn = (request, next) => {
  const session = inject(AuthSession);
  const router = inject(Router);
  const token = session.getToken();
  const isBackendRequest = request.url.startsWith(environment.apiUrl);

  if (!token || !isBackendRequest) {
    return next(request);
  }

  return next(
    request.clone({
      setHeaders: {
        Authorization: `Bearer ${token}`,
      },
    }),
  ).pipe(
    catchError((error) => {
      if (error?.status === 401) {
        session.clearSession();
        void router.navigate(['/login']);
      }

      return throwError(() => error);
    }),
  );
};
