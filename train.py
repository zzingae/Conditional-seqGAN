import torch
from torch.utils.tensorboard import SummaryWriter
import os
from model import Source_Generator, make_model
from utils import *
# from nltk.translate.bleu_score import sentence_bleu
import argparse
import torch.optim as optim


def write_summary(writer, prefix, values, step):
    for key, value in values.items():
        writer.add_scalar(prefix+'/'+key, value, step)

def evaluation(generator, oracle, test_sources, args):
    oracle_nll = 0
    count = 0
    loss_fn = nn.NLLLoss() # By default, the losses are averaged over each loss element in the batch
    with torch.no_grad():
        for i in range(0, len(test_sources), args.batch_size):
            batch_sources = test_sources[i:i+args.batch_size]

            memory = generator.encode(batch_sources, None)
            batch_samples, _ = beam_decode(generator, memory, None, args)

            batch_target_xs = batch_samples[:,:-1]
            batch_target_ys = batch_samples[:,1:]
            sample_mask = Batch.make_std_mask(batch_target_xs,pad=-1) # -1 for ignoring pad argument

            logits = oracle(batch_sources, batch_target_xs, None, sample_mask)
            logits = oracle.generator(logits)

            oracle_nll += (loss_fn(logits.flatten(0,1), batch_target_ys.flatten(0,1)) / args.max_target_len)
            count += 1

    return oracle_nll / count

def train_generator_MLE(generator, oracle, oracle_sources, oracle_targets, test_sources, criterion, opti, args, oracle_nll_step):
    step=0
    for epoch in range(args.mle_epochs):
        indices = torch.randperm(len(oracle_sources))
        # shuffle training data
        train_sources = oracle_sources[indices]
        train_targets = oracle_targets[indices]
        for i in range(0, len(oracle_targets), args.batch_size):
            batch_sources = train_sources[i:i+args.batch_size]
            batch_target_xs = train_targets[i:i+args.batch_size, :-1]
            batch_target_ys = train_targets[i:i+args.batch_size, 1:]

            # Clear gradients
            opti.optimizer.zero_grad()

            target_mask = Batch.make_std_mask(batch_target_xs, pad=-1) # -1 for ignoring pad argument
            logits = generator(batch_sources, batch_target_xs, None, target_mask)
            logits = generator.generator(logits)

            ntokens = batch_target_ys.shape[0]*batch_target_ys.shape[1]
            loss = criterion(logits, batch_target_ys) / ntokens
            loss.backward()

            #Optimization step
            opti.step()

            if (step + 1) % 10 == 0:
                write_summary(writer, 'generator/MLE', {'loss': loss,'lr': opti._rate}, step+1)
                print("Iteration %d of epoch %d complete. Loss : %.4f" % (step+1, epoch+1, loss))
            step += 1

        generator.eval()
        oracle_nll = evaluation(generator, oracle, test_sources, args)
        generator.train()

        write_summary(writer, 'generator/eval', {'oracle_nll': oracle_nll}, oracle_nll_step)
        print("Evaluation %d complete. oracle_nll : %.4f" % (epoch+1, oracle_nll))
        oracle_nll_step += 1
    return oracle_nll_step

def train_generator_PG(generator, discriminator, batch_sources, gen_optimizer, args):
    # generator training
    gen_optimizer.zero_grad()

    with torch.no_grad():
        memory = generator.encode(batch_sources, None)
        batch_targets, _ = beam_decode(generator, memory, None, args)
        # save rewards for every batch x sequence tokens
        rewards = torch.zeros((len(batch_targets), args.max_target_len)).to(args.device)

        memory = generator.encode(batch_sources, None)
        for l in range(1,args.max_target_len+1): # starting from 1 for ignoring 'start' token
            if l != args.max_target_len:
                _, monte_searchs = beam_decode(generator, memory, None, args, tokens=batch_targets[:,:l+1])
            else:
                monte_searchs = batch_targets # for entire sequence reward

            search_num = int(len(monte_searchs)/len(batch_sources))
            batch_sources_expand = batch_sources.repeat_interleave(search_num, dim=0)
            dis_probs = discriminator(batch_sources_expand, monte_searchs, None, None)
            rewards[:,l-1] = torch.mean(dis_probs.view(-1, search_num), dim=1) # mean of searchs is reward

    batch_target_xs = batch_targets[:,:-1]
    batch_target_ys = batch_targets[:,1:]

    target_mask = Batch.make_std_mask(batch_target_xs, pad=-1) # -1 for ignoring pad argument
    logits = generator(batch_sources, batch_target_xs, None, target_mask)
    logits = generator.generator(logits)

    log_prob = torch.gather(logits, dim=2, index=batch_target_ys.unsqueeze(-1))
    rewards = rewards.unsqueeze(-1)
    ntokens = batch_target_ys.shape[0]*batch_target_ys.shape[1]

    # minimize negative log probability with reward (policy gradient)
    pg_loss = -torch.sum(log_prob * rewards) / ntokens
    pg_loss.backward()
    gen_optimizer.step()

    return pg_loss, rewards


def train_discriminator(generator, discriminator, oracle_sources, oracle_targets, opti, args, d_steps, dis_epochs, discriminator_step):
    for d_step in range(d_steps):
        print('d-step: {}'.format(d_step+1))
        # generate negative samples
        neg_train = []
        for i in range(0, len(oracle_sources), args.batch_size):
            batch_sources = oracle_sources[i:i+args.batch_size]
            memory = generator.encode(batch_sources, None)
            batch_targets, _ = beam_decode(generator, memory, None, args)

            neg_train.append(batch_targets)

        neg_train = torch.cat(neg_train,dim=0)
        pos_train = oracle_targets # positive samples are oracle samples

        # permutation
        indices = torch.randperm(len(oracle_sources))
        oracle_sources_permute = oracle_sources[indices]
        neg_train = neg_train[indices]
        pos_train = pos_train[indices]

        for epoch in range(dis_epochs):
            total_loss = 0
            total_acc = 0
            count = 0
            for i in range(0, len(oracle_sources_permute), args.batch_size):
                batch_sources = oracle_sources_permute[i:i+args.batch_size]
                batch_positive = pos_train[i:i+args.batch_size]
                batch_negative = neg_train[i:i+args.batch_size]

                dis_sources = batch_sources.repeat(2,1) # sources are the same for pos,neg
                dis_targets = torch.cat((batch_positive, batch_negative),dim=0)
                dis_labels = torch.ones(len(dis_targets)).float().to(args.device)
                dis_labels[len(batch_positive):] = 0 # 1 for pos, 0 for neg

                # Clear gradients
                opti.zero_grad()

                dis_probs = discriminator(dis_sources, dis_targets, None, None) # discriminate pos,neg targets given sources

                loss_fn = nn.BCELoss()
                loss = loss_fn(dis_probs, dis_labels)
                loss.backward()
                opti.step()

                acc = torch.mean(((dis_probs>0.5)==(dis_labels>0.5)).float())

                total_loss += loss
                total_acc += acc
                count += 1

            print('discriminator average_loss = %.4f, average_accuracy = %.4f' % (total_loss/count, total_acc/count))
            write_summary(writer, 'discriminator/', {'loss': total_loss/count, 'acc': total_acc/count}, discriminator_step)
            discriminator_step += 1
    return discriminator_step

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--max_source_len', type=int, default=10)
    parser.add_argument('--max_target_len', type=int, default=10)
    parser.add_argument('--vocab_size', type=int, default=1000)
    parser.add_argument('--embed_size', type=int, default=128)

    parser.add_argument('--warmup_steps', type=int, default=800)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--mle_lr', type=float, default=2)
    parser.add_argument('--gen_lr', type=float, default=0.001)
    parser.add_argument('--dis_lr', type=float, default=0.01)

    parser.add_argument('--mle_epochs', type=int, default=20)
    parser.add_argument('--gan_gen_batchs', type=int, default=10)
    parser.add_argument('--pre_d_steps', type=int, default=50)
    parser.add_argument('--gan_d_steps', type=int, default=5)
    parser.add_argument('--pre_dis_epochs', type=int, default=3)
    parser.add_argument('--gan_dis_epochs', type=int, default=3)
    parser.add_argument('--gan_epoch', type=int, default=50)

    parser.add_argument('--beam_width', type=int, default=10) # beam search for evaluation and monte carlo search
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--pos_neg_samples', type=int, default=10000)
    parser.add_argument('--oracle_var', type=float, default=1)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: {}".format(device))
    args.device = device
    num_gpus = torch.cuda.device_count()

    fix_random(random_seed=0)

    if not os.path.exists('./output'):
        os.mkdir('./output')

    oracle, gen, dis = make_model(vocab_size=args.vocab_size, d_model=args.embed_size, N=args.num_layers)
    source_generator = Source_Generator(args.embed_size,args.embed_size,args.vocab_size,args.max_source_len,
                        gpu=(device.type=='cuda'),oracle_init=True)

    gen = gen.to(device)
    dis = dis.to(device)

    source_generator = source_generator.to(device)
    oracle = oracle.to(device)
    source_generator.eval()
    oracle.eval()

    # generate train sources/targets
    oracle_sources = []
    oracle_targets = []
    for i in range(int(args.pos_neg_samples/args.batch_size)):
        sources = source_generator.sample(args.batch_size)
        memory = oracle.encode(sources, None)
        targets, _ = beam_decode(oracle, memory, None, args)

        oracle_sources.append(sources)
        oracle_targets.append(targets)

    oracle_sources = torch.cat(oracle_sources,dim=0)
    oracle_targets = torch.cat(oracle_targets,dim=0)
    print('generated oracle source/target samples {}/{} for training'.format(len(oracle_sources),len(oracle_targets)))

    # generate test sources for evaluation
    test_sources = torch.cat([source_generator.sample(args.batch_size) for i in range(int(args.pos_neg_samples/args.batch_size))],dim=0)
    print('generated oracle source samples {} for testing'.format(len(test_sources)))

    writer = SummaryWriter('./output')
    oracle_nll_step = 1
    discriminator_step = 1

    criterion = LabelSmoothing(args.vocab_size, smoothing=args.label_smoothing)
    opti = get_my_opt(gen, learning_rate=args.mle_lr, warmup_steps=args.warmup_steps)
    # pre-train generator using MLE
    oracle_nll_step = train_generator_MLE(gen, oracle, oracle_sources, oracle_targets, test_sources, criterion, opti, args, oracle_nll_step)

    if num_gpus>1:
        dis = torch.nn.DataParallel(dis)
        dis_optimizer = optim.Adagrad(dis.module.parameters(),lr=args.dis_lr)
        print('use {} gpus for discriminator training'.format(num_gpus))
    else:
        dis_optimizer = optim.Adagrad(dis.parameters(),lr=args.dis_lr)

    # pre-train discriminator
    gen.eval()
    discriminator_step = train_discriminator(gen, dis, oracle_sources, oracle_targets, dis_optimizer, args, args.pre_d_steps, args.pre_dis_epochs, discriminator_step)

    # training generator and discriminator alternatively
    gen_optimizer = optim.Adam(gen.parameters(), lr=args.gen_lr)
    gen_steps = 0
    for epoch in range(args.gan_epoch):
        print('\n--------\nGAN EPOCH %d\n--------' % (epoch+1))
        indices = torch.randperm(len(oracle_sources))
        train_sources = oracle_sources[indices]
        for i in range(0, len(oracle_targets), args.batch_size):
            batch_sources = train_sources[i:i+args.batch_size]

            gen.train()
            dis.eval()
            pg_loss, rewards = train_generator_PG(gen, dis, batch_sources, gen_optimizer, args)

            write_summary(writer, 'generator/GAN', {'loss': pg_loss, 'reward': torch.mean(rewards)}, gen_steps)
            print("Generator, Loss : %.4f Reward : %.4f" % (pg_loss, torch.mean(rewards)))
            gen_steps+=1

            if gen_steps % args.gan_gen_batchs == 0:
                gen.eval()
                dis.train()
                discriminator_step = train_discriminator(gen, dis, oracle_sources, oracle_targets, dis_optimizer, args, args.gan_d_steps, args.gan_dis_epochs, discriminator_step)

        gen.eval()
        oracle_nll = evaluation(gen, oracle, test_sources, args)
        
        write_summary(writer, 'generator/eval', {'oracle_nll': oracle_nll}, oracle_nll_step)
        print("Evaluation epoch %d complete. oracle_nll : %.4f" % (oracle_nll_step, oracle_nll))
        oracle_nll_step += 1