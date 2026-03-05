"use client";

import { 
  Button, 
  Card, 
  CardHeader, 
  CardTitle, 
  CardContent, 
  CardFooter,
  Badge
} from "@alpha/ui";
import { 
  Activity, 
  Cpu, 
  Zap, 
  Shield, 
  AlertTriangle, 
  CheckCircle2,
  Download,
  Terminal,
  Play,
  Settings
} from "lucide-react";

export default function StyleGuidePage() {
  return (
    <div className="container mx-auto max-w-5xl px-4 py-12">
      <header className="mb-12 border-b border-border pb-8">
        <h1 className="text-4xl font-bold tracking-tight text-white mb-2">Alpha Style Guide</h1>
        <p className="text-text-secondary text-lg">
          Core UI components for the Alpha training system. All components are built with React 19, Tailwind CSS v4, and Lucide Icons.
        </p>
      </header>

      <section className="mb-16">
        <h2 className="text-2xl font-semibold text-white mb-6 border-l-4 border-accent pl-4">Buttons</h2>
        <div className="grid gap-8">
          <Card>
            <CardHeader>
              <CardTitle>Variants</CardTitle>
            </CardHeader>
            <CardContent className="flex flex-wrap gap-4">
              <Button variant="primary">Primary Button</Button>
              <Button variant="secondary">Secondary Button</Button>
              <Button variant="outline">Outline Button</Button>
              <Button variant="ghost">Ghost Button</Button>
              <Button variant="danger">Danger Button</Button>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Sizes</CardTitle>
            </CardHeader>
            <CardContent className="flex flex-wrap items-end gap-4">
              <Button size="sm">Small</Button>
              <Button size="md">Medium</Button>
              <Button size="lg">Large</Button>
              <Button size="icon" variant="outline"><Settings className="h-4 w-4" /></Button>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>States</CardTitle>
            </CardHeader>
            <CardContent className="flex flex-wrap gap-4">
              <Button disabled>Disabled Button</Button>
              <Button variant="outline" className="gap-2">
                <Download className="h-4 w-4" />
                With Icon
              </Button>
              <Button variant="primary" className="gap-2">
                <Play className="h-4 w-4" />
                Start Training
              </Button>
            </CardContent>
          </Card>
        </div>
      </section>

      <section className="mb-16">
        <h2 className="text-2xl font-semibold text-white mb-6 border-l-4 border-green pl-4">Badges</h2>
        <Card>
          <CardHeader>
            <CardTitle>Status & Categorization</CardTitle>
          </CardHeader>
          <CardContent className="flex flex-wrap gap-4">
            <Badge variant="default">Default</Badge>
            <Badge variant="secondary">Secondary</Badge>
            <Badge variant="outline">Outline</Badge>
            <Badge variant="success" className="gap-1">
              <CheckCircle2 className="h-3 w-3" />
              Completed
            </Badge>
            <Badge variant="warning" className="gap-1">
              <AlertTriangle className="h-3 w-3" />
              Stale
            </Badge>
            <Badge variant="danger" className="gap-1">
              <Shield className="h-3 w-3" />
              Failed
            </Badge>
            <Badge variant="blue" className="gap-1">
              <Activity className="h-3 w-3" />
              Active
            </Badge>
          </CardContent>
        </Card>
      </section>

      <section className="mb-16">
        <h2 className="text-2xl font-semibold text-white mb-6 border-l-4 border-purple-400 pl-4">Cards & Layout</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Cpu className="h-4 w-4 text-accent" />
                System Status
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-text-secondary leading-relaxed">
                Helios engine is currently utilizing 85% of GPU VRAM. Training throughput is stable at 12k tokens/sec.
              </p>
            </CardContent>
            <CardFooter className="flex justify-between border-t border-border mt-4 pt-4">
              <span className="text-[0.6rem] text-text-muted font-mono uppercase">L4 Tensor Core</span>
              <Badge variant="success">Healthy</Badge>
            </CardFooter>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Zap className="h-4 w-4 text-yellow" />
                Evolutionary Search
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-text-secondary leading-relaxed">
                Symbiogenesis generation 12 complete. Discovered 3 potential candidates with better loss than GELU.
              </p>
            </CardContent>
            <CardFooter className="flex justify-between border-t border-border mt-4 pt-4">
              <span className="text-[0.6rem] text-text-muted font-mono uppercase">G-Alpha.1</span>
              <Button size="sm" variant="outline">Analyze</Button>
            </CardFooter>
          </Card>

          <Card className="border-accent/30 bg-accent/5">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-accent">
                <Terminal className="h-4 w-4" />
                CLI Deployment
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-text-secondary leading-relaxed">
                Quickly deploy Alpha models to edge devices using our pre-compiled SPIR-V binary outputs.
              </p>
            </CardContent>
            <CardFooter className="flex justify-end mt-4 pt-4">
              <Button size="sm">Deploy Now</Button>
            </CardFooter>
          </Card>
        </div>
      </section>

      <section className="mb-16">
        <h2 className="text-2xl font-semibold text-white mb-6 border-l-4 border-yellow pl-4">Typography & Colors</h2>
        <Card>
          <CardContent className="grid grid-cols-1 md:grid-cols-2 gap-12 pt-6">
            <div className="space-y-4">
              <h3 className="text-text-muted uppercase text-[0.65rem] font-bold tracking-widest">Colors</h3>
              <div className="grid grid-cols-4 gap-4">
                <div className="space-y-2">
                  <div className="h-12 w-full rounded-md bg-accent border border-white/10" />
                  <span className="text-[0.6rem] text-text-muted font-mono">accent</span>
                </div>
                <div className="space-y-2">
                  <div className="h-12 w-full rounded-md bg-green border border-white/10" />
                  <span className="text-[0.6rem] text-text-muted font-mono">green</span>
                </div>
                <div className="space-y-2">
                  <div className="h-12 w-full rounded-md bg-yellow border border-white/10" />
                  <span className="text-[0.6rem] text-text-muted font-mono">yellow</span>
                </div>
                <div className="space-y-2">
                  <div className="h-12 w-full rounded-md bg-red border border-white/10" />
                  <span className="text-[0.6rem] text-text-muted font-mono">red</span>
                </div>
                <div className="space-y-2">
                  <div className="h-12 w-full rounded-md bg-surface border border-white/10" />
                  <span className="text-[0.6rem] text-text-muted font-mono">surface</span>
                </div>
                <div className="space-y-2">
                  <div className="h-12 w-full rounded-md bg-surface-2 border border-white/10" />
                  <span className="text-[0.6rem] text-text-muted font-mono">surface-2</span>
                </div>
                <div className="space-y-2">
                  <div className="h-12 w-full rounded-md bg-border border border-white/10" />
                  <span className="text-[0.6rem] text-text-muted font-mono">border</span>
                </div>
                <div className="space-y-2">
                  <div className="h-12 w-full rounded-md bg-border-2 border border-white/10" />
                  <span className="text-[0.6rem] text-text-muted font-mono">border-2</span>
                </div>
              </div>
            </div>

            <div className="space-y-4">
              <h3 className="text-text-muted uppercase text-[0.65rem] font-bold tracking-widest">Text Hierarchy</h3>
              <div className="space-y-4">
                <div>
                  <div className="text-white text-3xl font-bold">Heading 1</div>
                  <span className="text-[0.6rem] text-text-muted font-mono">3xl / bold / white</span>
                </div>
                <div>
                  <div className="text-text-primary text-xl font-semibold">Heading 2</div>
                  <span className="text-[0.6rem] text-text-muted font-mono">xl / semibold / text-primary</span>
                </div>
                <div>
                  <div className="text-text-secondary text-base">Standard paragraph text.</div>
                  <span className="text-[0.6rem] text-text-muted font-mono">base / text-secondary</span>
                </div>
                <div>
                  <div className="text-text-muted text-sm font-mono uppercase tracking-widest">Label Text</div>
                  <span className="text-[0.6rem] text-text-muted font-mono">sm / mono / uppercase / text-muted</span>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </section>
    </div>
  );
}
