import { Component } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { HttpClient, HttpClientModule } from '@angular/common/http';
import json5 from 'json5';
import * as Prism from 'prismjs';
import 'prismjs/components/prism-python';
import 'prismjs/themes/prism-okaidia.css';
enum DifficultyLevel {
  EASY = "easy",
  MEDIUM = "medium",
  HARD = "hard"
}

enum ProblemTopic {
  ARRAYS = "arrays",
  STRINGS = "strings",
  LINKED_LISTS = "linked_lists",
  TREES = "trees",
  GRAPHS = "graphs",
  DYNAMIC_PROGRAMMING = "dynamic_programming",
  SORTING = "sorting",
  SEARCHING = "searching",
  RECURSION = "recursion",
  BACKTRACKING = "backtracking"
}

interface ProblemExample {
  input: string;
  output: string;
  explanation?: string;
}

interface ProblemResponse {
  id: string;
  title: string;
  description: string;
  constraints: string[];
  examples: ProblemExample[];
  difficulty: DifficultyLevel;
  topic: ProblemTopic;
  solution?: string; // This might be added by the API or we may need to request it separately
}

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [RouterOutlet, CommonModule, FormsModule, HttpClientModule],
  templateUrl: './app.html',
  styleUrl: './app.css'
})
export class App {
  topic: string = 'arrays';
  difficulty: string = 'medium';
  optionalPrompt: string = '';
  isLoading: boolean = false;
  error: string | null = null;
  challenge: ProblemResponse | null = null;

  constructor(private http: HttpClient) {}
  //parsedData: ParsedData | null = null;
  parseError: string | null = null;


  generateChallenge(event: Event): void {
    event.preventDefault();
    this.isLoading = true;
    this.error = null;
    this.challenge = null;

    // Make the API request with the proper format
    this.http.post<any>('http://127.0.0.1:8000/api/problems/verified', {
      topic: this.topic as ProblemTopic,
      difficulty: this.difficulty as DifficultyLevel,
      user_prompt: this.optionalPrompt
    }).subscribe({
      next: (response) => {
        this.isLoading = false;
        try {
          // Parse the response - using json5 for more robust parsing
          const parsedResponse = typeof response.response === 'string'
            ? json5.parse(response.response)
            : response.response;

          this.challenge = parsedResponse;
          console.log(parsedResponse);
          // If the API doesn't provide a python_solution, we might need to handle that case
          if (!this.challenge?.solution) {
            console.warn('No Python solution provided in the response');
          }
        } catch (err) {
          this.error = 'Failed to parse the response. Please try again.';
          console.error('Error parsing response:', err);
        }
      },
      error: (err) => {
        this.isLoading = false;
        this.error = 'Failed to connect to the server. Please check your connection and try again.';
        console.error('API error:', err);
      }
    });
  }
  get highlightedCode() {
    return this.challenge?.solution ?
      Prism.highlight(this.challenge.solution, Prism.languages["python"], 'python') : '';
  }
  copySolution(): void {
    if (this.challenge?.solution) {
      navigator.clipboard.writeText(this.challenge.solution)
        .then(() => {
          alert('Solution code copied to clipboard!');
        })
        .catch(err => {
          console.error('Failed to copy text:', err);
        });
    }
  }
}
