import React, { useState, useEffect } from 'react';
import { Navigate, useNavigate, useLocation } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { ShieldCheckIcon } from '@heroicons/react/24/outline';

const Login = () => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const navigate = useNavigate();
  const location = useLocation();
  const { login, isAuthenticated } = useAuth();
  
  const from = location.state?.from?.pathname || '/';
  
  useEffect(() => {
    if (isAuthenticated) {
      navigate(from, { replace: true });
    }
  }, [isAuthenticated, navigate, from]);

  if (isAuthenticated) {
    return <Navigate to="/" replace />;
  }

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setIsLoading(true);

    try {
      // For demo, using hardcoded credentials
      if (username === 'demo' && password === 'demo123') {
        await login({ username, password });
        navigate(from, { replace: true });
      } else if (username === 'abc' && password === 'abc') {
        // Backward compatibility
        await login({ username, password });
        navigate(from, { replace: true });
      } else {
        setError('Invalid username or password');
      }
    } catch (err) {
      setError(err.message || 'An error occurred during login. Please try again.');
      console.error('Login error:', err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-800 to-indigo-900 flex items-center justify-center p-4">
      <div className="w-full max-w-md">
        <div className="bg-white/90 rounded-2xl shadow-2xl overflow-hidden border border-gray-200">
          <div className="p-10">
            <div className="text-center mb-10">
              <div className="mx-auto h-16 w-16 bg-blue-100 rounded-2xl flex items-center justify-center mb-4">
                <ShieldCheckIcon className="h-8 w-8 text-blue-600" />
              </div>  
              <h1 className="text-3xl font-bold text-gray-800 mb-2">E-ZenithCare</h1>
              <h2 className="text-3xl font-bold text-gray-800 mb-2">Welcomes you</h2>
              <p className="text-gray-600">Sign in to continue</p>
            </div>
            
            {error && (
              <div className="mb-6 p-4 bg-red-500/10 border border-red-500/30 text-red-100 rounded-xl text-sm flex items-center">
                <svg className="w-5 h-5 mr-2 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                </svg>
                {error}
              </div>
            )}
            
            <form onSubmit={handleSubmit} className="space-y-6">
              <div className="space-y-2">
                <label htmlFor="username" className="text-sm font-medium text-gray-700 block">
                  Username
                </label>
                <div className="relative">
                  <input
                    id="username"
                    name="username"
                    type="text"
                    required
                    value={username}
                    onChange={(e) => setUsername(e.target.value)}
                    className="block w-full px-4 py-3 bg-white border border-gray-300 text-gray-900 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all duration-200"
                    placeholder="Enter username"
                  />
                </div>
              </div>

              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <label htmlFor="password" className="text-sm font-medium text-gray-700 block">
                    Password
                  </label>
                  <div className="text-sm">
                    <button 
                      type="button" 
                      className="font-medium text-blue-600 hover:text-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
                      onClick={() => {/* TODO: Implement password reset */}}
                    >
                      Forgot your password?
                    </button>
                  </div>
                </div>
                <div className="relative">
                  <input
                    id="password"
                    name="password"
                    type="password"
                    required
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    className="block w-full px-4 py-3 bg-white border border-gray-300 text-gray-900 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all duration-200"
                    placeholder="Enter password"
                  />
                </div>
              </div>

              <div className="flex items-center">
                <input
                  id="remember-me"
                  name="remember-me"
                  type="checkbox"
                  className="h-4 w-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
                />
                <label htmlFor="remember-me" className="ml-2 block text-sm text-gray-700">
                  Remember this device
                </label>
              </div>

              <div>
                <button
                  type="submit"
                  disabled={isLoading}
                  className={`w-full flex justify-center items-center py-3.5 px-4 border border-transparent rounded-lg text-sm font-semibold text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 transition-all duration-200 ${isLoading ? 'opacity-80 cursor-not-allowed' : ''}`}
                >
                  {isLoading ? (
                    <>
                      <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                      </svg>
                      Signing in...
                    </>
                  ) : 'Sign in'}
                </button>
              </div>
            </form>
          </div>
          
          <div className="px-8 py-5 bg-white/5 border-t border-white/10 text-center">
            <p className="text-sm text-blue-100/70">
              Don't have an account?{' '}
            </p>
          </div>
        </div>

        <div className="mt-6 text-center">
          <p className="text-xs text-blue-200/60">
            {new Date().getFullYear()} E-ZenithCare. All rights reserved.
          </p>
        </div>
      </div>
    </div>
  );
};

export default Login;
