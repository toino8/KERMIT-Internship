// app/page.tsx
import { redirect } from 'next/navigation';

export default function Home() {
  // Redirection vers la page "About"
  redirect('/home');
  return null;
}
